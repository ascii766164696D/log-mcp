from mcp.server.fastmcp import FastMCP

mcp = FastMCP("log-analysis")

# Register all tools
from .tools import classify, compare, errors, overview, search, segment, stats

overview.register_tools(mcp)
search.register_tools(mcp)
segment.register_tools(mcp)
errors.register_tools(mcp)
stats.register_tools(mcp)
compare.register_tools(mcp)
classify.register_tools(mcp)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
