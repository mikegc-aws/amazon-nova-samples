import os 
import asyncio
import base64
import json
import uuid
import warnings
import pyaudio
import pytz
import random
import hashlib
import datetime
import time
import inspect
import numpy as np
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
import argparse
# MCP imports for stdio support only
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

# Import LiveKit's AudioProcessingModule for echo cancellation
try:
    from livekit import rtc
    APM_AVAILABLE = True
except ImportError:
    APM_AVAILABLE = False
    print("Warning: LiveKit RTC not available. Echo cancellation will be disabled.")
    print("Install with: pip install livekit")

# Suppress warnings
warnings.filterwarnings("ignore")

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of frames per buffer

# Debug mode flag
DEBUG = False

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        functionName = inspect.stack()[1].function
        if  functionName == 'time_it' or functionName == 'time_it_async':
            functionName = inspect.stack()[2].function
        print('{:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())[:-3] + ' ' + functionName + ' ' + message)

def time_it(label, methodToRun):
    start_time = time.perf_counter()
    result = methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

async def time_it_async(label, methodToRun):
    start_time = time.perf_counter()
    result = await methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

class MCPManager:
    """Manages connections to one or more MCP stdio servers, and provides tool listing and invocation."""
    def __init__(self, mcp_configs):
        self.mcp_configs = mcp_configs
        self.exit_stack = AsyncExitStack()
        self.sessions = []  # List of (session, server_type, config)

    async def __aenter__(self):
        for cfg in self.mcp_configs:
            if cfg['type'] == 'stdio':
                params = StdioServerParameters(**cfg['params'])
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                self.sessions.append((session, 'stdio', cfg))
            else:
                raise ValueError(f"Unknown MCP server type: {cfg['type']}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def list_tools(self):
        all_tools = []
        for session, server_type, cfg in self.sessions:
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"[DEBUG] MCP server tool: name={tool.name}, schema={tool.inputSchema}")
                all_tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema,
                    'server_type': server_type,
                    'session': session,
                    'server_cfg': cfg
                })
        return all_tools

    async def call_tool(self, tool_name, tool_input):
        for session, server_type, cfg in self.sessions:
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                if tool.name == tool_name:
                    print(f"[DEBUG] MCPManager.call_tool: tool_name={tool_name}, tool_input={tool_input}")
                    response = await session.call_tool(tool_name, tool_input)
                    print(f"[DEBUG] Raw response from MCP server: {response}")
                    if hasattr(response, 'content'):
                        return [c.text for c in response.content]
                    return response
        raise ValueError(f"Tool {tool_name} not found on any MCP server.")

class BedrockStreamManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""
    
    # Event templates
    START_SESSION_EVENT = '''{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
                }
            }
        }
    }'''

    CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }'''

    AUDIO_EVENT_TEMPLATE = '''{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TEXT_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": true,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
        }
    }'''

    TEXT_INPUT_EVENT = '''{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TOOL_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }'''

    CONTENT_END_EVENT = '''{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }'''

    PROMPT_END_EVENT = '''{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }'''

    SESSION_END_EVENT = '''{
        "event": {
            "sessionEnd": {}
        }
    }'''
    
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1', mcp_manager=None):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        self.mcp_manager = mcp_manager
        
        # Replace RxPy subjects with asyncio queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.barge_in = False
        self.bedrock_client = None
        
        # Audio playback components
        self.audio_player = None
        
        # Text response components
        self.display_assistant_text = False
        self.role = None

        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        self.dynamic_tools = []

    async def load_tools_from_mcp(self):
        if self.mcp_manager:
            self.dynamic_tools = await self.mcp_manager.list_tools()

    def start_prompt(self):
        """Create a promptStart event using dynamic tools from MCP if available."""
        tools = []
        if self.dynamic_tools:
            for tool in self.dynamic_tools:
                tools.append({
                    "toolSpec": {
                        "name": tool['name'],
                        "description": tool['description'],
                        "inputSchema": {"json": json.dumps(tool['inputSchema'])}
                    }
                })
        else:
            # fallback to hardcoded tools if no MCP tools loaded
            get_default_tool_schema = json.dumps({
                "type": "object",
                "properties": {},
                "required": []
            })
            get_order_tracking_schema = json.dumps({
                "type": "object",
                "properties": {
                    "orderId": {
                        "type": "string",
                        "description": "The order number or ID to track"
                    },
                    "requestNotifications": {
                        "type": "boolean",
                        "description": "Whether to set up notifications for this order",
                        "default": False
                    }
                },
                "required": ["orderId"]
            })
            tools = [
                {
                    "toolSpec": {
                        "name": "getDateAndTimeTool",
                        "description": "get information about the current date and time",
                        "inputSchema": {"json": get_default_tool_schema}
                    }
                },
                {
                    "toolSpec": {
                        "name": "trackOrderTool",
                        "description": "Retrieves real-time order tracking information and detailed status updates for customer orders by order ID. Provides estimated delivery dates. Use this tool when customers ask about their order status or delivery timeline.",
                        "inputSchema": {"json": get_order_tracking_schema}
                    }
                }
            ]
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": tools
                    }
                }
            }
        }
        return json.dumps(prompt_start_event)
    
    def tool_result_event(self, content_name, content, role):
        """Create a tool result event"""
        # Do NOT json.dumps the content if it's a dict; include it directly
        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content  # pass as dict or string, not stringified JSON
                }
            }
        }
        return json.dumps(tool_result_event)
   
    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        if not self.bedrock_client:
            self._initialize_client()
        
        try:
            self.stream_response = await time_it_async("invoke_model_with_bidirectional_stream", lambda : self.bedrock_client.invoke_model_with_bidirectional_stream( InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)))
            self.is_active = True
            default_system_prompt = "You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation." \
            "When reading order numbers, please read each digit individually, separated by pauses. For example, order #1234 should be read as 'order number one-two-three-four' rather than 'order number one thousand two hundred thirty-four'."
            
            # Send initialization events
            prompt_event = self.start_prompt()
            text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM")
            text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt)
            text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
            
            init_events = [self.START_SESSION_EVENT, prompt_event, text_content_start, text_content, text_content_end]
            
            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events
                await asyncio.sleep(0.1)
            
            # Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start processing audio input
            asyncio.create_task(self._process_audio_input())
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            debug_print("Stream initialized successfully")
            return self
        except Exception as e:
            self.is_active = False
            print(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            debug_print("Stream not initialized or closed")
            return
       
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            # For debugging large events, you might want to log just the type
            if DEBUG:
                if len(event_json) > 200:
                    event_type = json.loads(event_json).get("event", {}).keys()
                    debug_print(f"Sent event type: {list(event_type)}")
                else:
                    debug_print(f"Sent event: {event_json}")
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def send_audio_content_start_event(self):
        """Send a content start event to the Bedrock stream."""
        content_start_event = self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_start_event)
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        while self.is_active:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    debug_print("No audio bytes received")
                    continue
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self.AUDIO_EVENT_TEMPLATE % (
                    self.prompt_name, 
                    self.audio_content_name, 
                    blob.decode('utf-8')
                )
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        self.audio_input_queue.put_nowait({
            'audio_bytes': audio_bytes,
            'prompt_name': self.prompt_name,
            'content_name': self.audio_content_name
        })
    
    async def send_audio_content_end_event(self):
        """Send a content end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_end_event)
        debug_print("Audio ended")
    
    async def send_tool_start_event(self, content_name):
        """Send a tool content start event to the Bedrock stream."""
        content_start_event = self.TOOL_CONTENT_START_EVENT % (self.prompt_name, content_name, self.toolUseId)
        debug_print(f"Sending tool start event: {content_start_event}")  
        await self.send_raw_event(content_start_event)

    async def send_tool_result_event(self, content_name, tool_result):
        print(f"[DEBUG] (pre-send-tool-result) self.is_active: {getattr(self, 'is_active', None)} self.stream_response: {getattr(self, 'stream_response', None)}")
        tool_result_event = self.tool_result_event(content_name=content_name, content=tool_result, role="TOOL")
        print(f"[DEBUG] Full tool_result_event JSON: {tool_result_event}")
        debug_print(f"Sending tool result event: {tool_result_event}")
        await self.send_raw_event(tool_result_event)
    
    async def send_tool_content_end_event(self, content_name):
        """Send a tool content end event to the Bedrock stream."""
        tool_content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, content_name)
        debug_print(f"Sending tool content event: {tool_content_end_event}")
        await self.send_raw_event(tool_content_end_event)
    
    async def send_prompt_end_event(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        prompt_end_event = self.PROMPT_END_EVENT % (self.prompt_name)
        await self.send_raw_event(prompt_end_event)
        debug_print("Prompt ended")
        
    async def send_session_end_event(self):
        """Send a session end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return

        await self.send_raw_event(self.SESSION_END_EVENT)
        self.is_active = False
        debug_print("Session ended")
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in json_data:
                                if 'contentStart' in json_data['event']:
                                    debug_print("Content start detected")
                                    content_start = json_data['event']['contentStart']
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                debug_print("Speculative content detected")
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            debug_print("Error parsing additionalModelFields")
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    role = json_data['event']['textOutput']['role']
                                    # Check if there is a barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        debug_print("Barge-in detected. Stopping audio output.")
                                        self.barge_in = True

                                    if (self.role == "ASSISTANT" and self.display_assistant_text):
                                        print(f"Assistant: {text_content}")
                                    elif (self.role == "USER"):
                                        print(f"User: {text_content}")

                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    await self.audio_output_queue.put(audio_bytes)
                                elif 'toolUse' in json_data['event']:
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                    debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")
                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    debug_print("Processing tool use and sending result")
                                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    print(f"[DEBUG] toolResult: {toolResult}")
                                    toolContent = str(uuid.uuid4())
                                    await self.send_tool_start_event(toolContent)
                                    await self.send_tool_result_event(toolContent, toolResult)
                                    await self.send_tool_content_end_event(toolContent)
                                
                                elif 'completionEnd' in json_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    print("End of response sequence")
                            
                            # Put the response in the output queue for other components
                            await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                   # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False

    async def processToolUse(self, toolName, toolUseContent):
        if self.mcp_manager:
            print(f"[DEBUG] (pre-tool-call) self.is_active: {getattr(self, 'is_active', None)} self.stream_response: {getattr(self, 'stream_response', None)}")
            content = toolUseContent.get("content", "{}")
            try:
                content_dict = json.loads(content)
            except Exception:
                content_dict = {}
            required = None
            if hasattr(self, 'dynamic_tools') and self.dynamic_tools:
                for tool in self.dynamic_tools:
                    if tool['name'] == toolName:
                        required = tool['inputSchema'].get('required', [])
                        print(f"[DEBUG] Required fields for '{toolName}': {required}")
            if required:
                trimmed_input = {k: v for k, v in content_dict.items() if k in required}
                print(f"[DEBUG] Sending only required fields: {trimmed_input}")
            else:
                trimmed_input = content_dict
            print(f"[DEBUG] About to call MCP tool '{toolName}' with input: {trimmed_input}")
            result = await self.mcp_manager.call_tool(toolName, trimmed_input)
            print(f"[DEBUG] (post-tool-call) self.is_active: {getattr(self, 'is_active', None)} self.stream_response: {getattr(self, 'stream_response', None)}")
            print(f"[DEBUG] MCP tool '{toolName}' result: {result}")
            print("-"*10)
            return json.dumps({"result": result})
        
        # fallback to old logic if no MCP
        tool = toolName.lower()
        debug_print(f"Tool Use Content: {toolUseContent}")
        if tool == "getdateandtimetool":
            pst_timezone = pytz.timezone("America/Los_Angeles")
            pst_date = datetime.datetime.now(pst_timezone)
            return {
                "formattedTime": pst_date.strftime("%I:%M %p"),
                "date": pst_date.strftime("%Y-%m-%d"),
                "year": pst_date.year,
                "month": pst_date.month,
                "day": pst_date.day,
                "dayOfWeek": pst_date.strftime("%A").upper(),
                "timezone": "PST"
            }
        elif tool == "trackordertool":
            content = toolUseContent.get("content", {})
            content_data = json.loads(content)
            order_id = content_data.get("orderId", "")
            request_notifications = toolUseContent.get("requestNotifications", False)
            if isinstance(order_id, int):
                order_id = str(order_id)
            if not order_id or not isinstance(order_id, str):
                return {
                    "error": "Invalid order ID format",
                    "orderStatus": "",
                    "estimatedDelivery": "",
                    "lastUpdate": ""
                }
            seed = int(hashlib.md5(order_id.encode(), usedforsecurity=False).hexdigest(), 16) % 10000
            random.seed(seed)
            statuses = [
                "Order received", "Processing", "Preparing for shipment",
                "Shipped", "In transit", "Out for delivery",
                "Delivered", "Delayed"
            ]
            weights = [10, 15, 15, 20, 20, 10, 5, 3]
            status = random.choices(statuses, weights=weights, k=1)[0]
            today = datetime.datetime.now()
            if status == "Delivered":
                delivery_days = -random.randint(0, 3)
                estimated_delivery = (today + datetime.timedelta(days=delivery_days)).strftime("%Y-%m-%d")
            elif status == "Out for delivery":
                estimated_delivery = today.strftime("%Y-%m-%d")
            else:
                delivery_days = random.randint(1, 10)
                estimated_delivery = (today + datetime.timedelta(days=delivery_days)).strftime("%Y-%m-%d")
            notification_message = ""
            if request_notifications and status != "Delivered":
                notification_message = f"You will receive notifications for order {order_id}"
            tracking_info = {
                "orderStatus": status,
                "orderNumber": order_id,
                "notificationStatus": notification_message
            }
            if status == "Delivered":
                tracking_info["deliveredOn"] = estimated_delivery
            elif status == "Out for delivery":
                tracking_info["expectedDelivery"] = "Today"
            else:
                tracking_info["estimatedDelivery"] = estimated_delivery
            if status == "In transit":
                tracking_info["currentLocation"] = "Distribution Center"
            elif status == "Delivered":
                tracking_info["deliveryLocation"] = "Front Door"
            if status == "Delayed":
                tracking_info["additionalInfo"] = "Weather delays possible"
            return tracking_info
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
       
        self.is_active = False
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

        await self.send_audio_content_end_event()
        await self.send_prompt_end_event()
        await self.send_session_end_event()

        if self.stream_response:
            await self.stream_response.input_stream.close()

class AudioStreamer:
    """Handles continuous microphone input and audio output using separate streams with echo cancellation."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.loop = asyncio.get_event_loop()

        # Initialize PyAudio
        debug_print("AudioStreamer Initializing PyAudio...")
        self.p = time_it("AudioStreamerInitPyAudio", pyaudio.PyAudio)
        debug_print("AudioStreamer PyAudio initialized")

        # Echo cancellation setup
        self.apm = None
        self.echo_cancellation_enabled = APM_AVAILABLE
        self.frame_duration_ms = 10  # WebRTC APM requires 10ms frames
        self.input_frames_per_10ms = int(INPUT_SAMPLE_RATE * self.frame_duration_ms / 1000)
        self.output_frames_per_10ms = int(OUTPUT_SAMPLE_RATE * self.frame_duration_ms / 1000)
        
        # Audio buffers for 10ms frame processing
        self.input_buffer = np.array([], dtype=np.int16)
        self.output_buffer = np.array([], dtype=np.int16)
        self.processed_output_buffer = np.array([], dtype=np.int16)
        
        # Timing for delay estimation
        self.last_output_time = 0
        self.last_input_time = 0
        
        if self.echo_cancellation_enabled:
            try:
                debug_print("Initializing WebRTC Audio Processing Module for echo cancellation...")
                self.apm = rtc.AudioProcessingModule(
                    echo_cancellation=True,
                    noise_suppression=True,
                    high_pass_filter=True,
                    auto_gain_control=True
                )
                # Set initial stream delay (can be adjusted dynamically)
                self.apm.set_stream_delay_ms(50)  # 50ms initial delay estimate
                debug_print("WebRTC APM initialized successfully")
            except Exception as e:
                debug_print(f"Failed to initialize WebRTC APM: {e}")
                self.echo_cancellation_enabled = False
                self.apm = None

        # Initialize separate streams for input and output
        # Input stream with callback for microphone
        debug_print("Opening input audio stream...")
        self.input_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.input_callback
        ))
        debug_print("input audio stream opened")

        # Output stream for direct writing (no callback)
        debug_print("Opening output audio stream...")
        self.output_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        ))

        debug_print("output audio stream opened")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Callback function that schedules audio processing in the asyncio event loop"""
        if self.is_streaming and in_data:
            self.last_input_time = time.time()
            # Schedule the task in the event loop
            asyncio.run_coroutine_threadsafe(
                self.process_input_audio(in_data), 
                self.loop
            )
        return (None, pyaudio.paContinue)

    async def process_input_audio(self, audio_data):
        """Process input audio with echo cancellation before sending to Bedrock"""
        try:
            if self.echo_cancellation_enabled and self.apm:
                # Convert audio data to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to buffer
                self.input_buffer = np.concatenate([self.input_buffer, audio_array])
                
                # Process in 10ms chunks
                processed_audio = b''
                while len(self.input_buffer) >= self.input_frames_per_10ms:
                    # Extract 10ms frame
                    frame_data = self.input_buffer[:self.input_frames_per_10ms]
                    self.input_buffer = self.input_buffer[self.input_frames_per_10ms:]
                    
                    # Create AudioFrame for APM processing
                    audio_frame = rtc.AudioFrame(
                        data=frame_data.tobytes(),
                        sample_rate=INPUT_SAMPLE_RATE,
                        num_channels=CHANNELS,
                        samples_per_channel=self.input_frames_per_10ms
                    )
                    
                    # Process the frame through APM (near-end processing)
                    self.apm.process_stream(audio_frame)
                    
                    # Append processed audio
                    processed_audio += audio_frame.data.tobytes()
                
                # Send processed audio if any
                if processed_audio:
                    self.stream_manager.add_audio_chunk(processed_audio)
            else:
                # Send audio directly if echo cancellation is not available
                self.stream_manager.add_audio_chunk(audio_data)
                
        except Exception as e:
            if self.is_streaming:
                print(f"Error processing input audio: {e}")

    def process_output_audio(self, audio_data):
        """Process output audio through APM for echo cancellation reference"""
        if not self.echo_cancellation_enabled or not self.apm:
            return audio_data
            
        try:
            # Convert to numpy array and resample if needed
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Add to buffer
            self.output_buffer = np.concatenate([self.output_buffer, audio_array])
            
            processed_audio = b''
            while len(self.output_buffer) >= self.output_frames_per_10ms:
                # Extract 10ms frame
                frame_data = self.output_buffer[:self.output_frames_per_10ms]
                self.output_buffer = self.output_buffer[self.output_frames_per_10ms:]
                
                # Create AudioFrame for APM processing
                audio_frame = rtc.AudioFrame(
                    data=frame_data.tobytes(),
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=CHANNELS,
                    samples_per_channel=self.output_frames_per_10ms
                )
                
                # Process the reverse stream (far-end processing for echo cancellation)
                self.apm.process_reverse_stream(audio_frame)
                
                # Append processed audio
                processed_audio += audio_frame.data.tobytes()
            
            return processed_audio if processed_audio else audio_data
            
        except Exception as e:
            debug_print(f"Error processing output audio for echo cancellation: {e}")
            return audio_data

    def update_stream_delay(self):
        """Update the stream delay for better echo cancellation"""
        if self.echo_cancellation_enabled and self.apm and self.last_output_time > 0 and self.last_input_time > 0:
            try:
                # Calculate delay between output and input processing
                delay_ms = int(abs(self.last_input_time - self.last_output_time) * 1000)
                # Clamp delay to reasonable bounds
                delay_ms = max(10, min(delay_ms, 500))
                self.apm.set_stream_delay_ms(delay_ms)
                debug_print(f"Updated APM stream delay to {delay_ms}ms")
            except Exception as e:
                debug_print(f"Error updating stream delay: {e}")
    
    async def play_output_audio(self):
        """Play audio responses from Nova Sonic with echo cancellation processing"""
        while self.is_streaming:
            try:
                # Check for barge-in flag
                if self.stream_manager.barge_in:
                    # Clear the audio queue
                    while not self.stream_manager.audio_output_queue.empty():
                        try:
                            self.stream_manager.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.stream_manager.barge_in = False
                    # Small sleep after clearing
                    await asyncio.sleep(0.05)
                    continue
                
                # Get audio data from the stream manager's queue
                audio_data = await asyncio.wait_for(
                    self.stream_manager.audio_output_queue.get(),
                    timeout=0.1
                )
                
                if audio_data and self.is_streaming:
                    self.last_output_time = time.time()
                    
                    # Process audio through APM for echo cancellation reference
                    processed_audio = self.process_output_audio(audio_data)
                    
                    # Update stream delay periodically
                    self.update_stream_delay()
                    
                    # Write the processed audio data in chunks
                    chunk_size = CHUNK_SIZE * 2  # bytes per sample (16-bit)
                    
                    # Write the audio data in chunks to avoid blocking too long
                    for i in range(0, len(processed_audio), chunk_size):
                        if not self.is_streaming:
                            break
                        
                        end = min(i + chunk_size, len(processed_audio))
                        chunk = processed_audio[i:end]
                        
                        # Create a new function that captures the chunk by value
                        def write_chunk(data):
                            return self.output_stream.write(data)
                        
                        # Pass the chunk to the function
                        await asyncio.get_event_loop().run_in_executor(None, write_chunk, chunk)
                        
                        # Brief yield to allow other tasks to run
                        await asyncio.sleep(0.001)
                    
            except asyncio.TimeoutError:
                # No data available within timeout, just continue
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error playing output audio: {str(e)}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start_streaming(self):
        """Start streaming audio."""
        if self.is_streaming:
            return
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        # Send audio content start event
        await time_it_async("send_audio_content_start_event", lambda : self.stream_manager.send_audio_content_start_event())
        
        self.is_streaming = True
        
        # Start the input stream if not already started
        if not self.input_stream.is_active():
            self.input_stream.start_stream()
        
        # Start processing tasks
        #self.input_task = asyncio.create_task(self.process_input_audio())
        self.output_task = asyncio.create_task(self.play_output_audio())
        
        # Wait for user to press Enter to stop
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Once input() returns, stop streaming
        await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False

        # Cancel the tasks
        tasks = []
        if hasattr(self, 'input_task') and not self.input_task.done():
            tasks.append(self.input_task)
        if hasattr(self, 'output_task') and not self.output_task.done():
            tasks.append(self.output_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Stop and close the streams
        if self.input_stream:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
        if self.p:
            self.p.terminate()
        
        await self.stream_manager.close() 


async def main(debug=False, disable_echo_cancellation=False, mcp_configs=None):
    global DEBUG, APM_AVAILABLE
    DEBUG = debug
    if disable_echo_cancellation or not APM_AVAILABLE:
        print("Echo cancellation disabled")
        APM_AVAILABLE = False
    if APM_AVAILABLE:
        print("WebRTC Audio Processing Module enabled for echo cancellation")
    else:
        print("Running without echo cancellation - speaker audio may be picked up by microphone")
    print("MCP Configs:", mcp_configs)  # Debug print
    async with MCPManager(mcp_configs or []) as mcp_manager:
        stream_manager = BedrockStreamManager(model_id='amazon.nova-sonic-v1:0', region='us-east-1', mcp_manager=mcp_manager)
        await stream_manager.load_tools_from_mcp()
        audio_streamer = AudioStreamer(stream_manager)
        await time_it_async("initialize_stream", stream_manager.initialize_stream)
        try:
            await audio_streamer.start_streaming()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            await audio_streamer.stop_streaming()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Nova Sonic Python Streaming with Tool Use and Echo Cancellation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--disable-echo-cancellation', action='store_true', help='Disable echo cancellation (useful for testing or if experiencing issues)')
    parser.add_argument('--mcp-server', action='append', help='MCP server config, e.g. type=stdio,command=uv,args=run;server.py,cwd=/path/to/dir', default=[])
    args = parser.parse_args()
    mcp_configs = []
    for s in args.mcp_server:
        cfg = {}
        for kv in s.split(','):
            k, v = kv.split('=', 1)
            if k == 'args':
                v = v.split(';') if v else []
            cfg[k] = v
        if cfg.get('type') == 'stdio':
            if 'args' in cfg and isinstance(cfg['args'], str):
                cfg['args'] = cfg['args'].split(';')
            mcp_configs.append({'type': 'stdio', 'params': cfg})
    try:
        asyncio.run(main(debug=args.debug, disable_echo_cancellation=args.disable_echo_cancellation, mcp_configs=mcp_configs))
    except Exception as e:
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
