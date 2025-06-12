uv run --with livekit nova_sonic_tool_use_mcp_task.py \
   --mcp-server "type=stdio,command=uv,args=run;main.py,cwd=/Users/danilop/Tests/Strands/Loom/strands-loom-mcp" \
   "$@"
