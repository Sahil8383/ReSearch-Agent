"""Main entry point for the AI agent application"""

from agent import ReActAgent


def main():
    """Main function to run the agent"""
    print("\n" + "🌟" * 40)
    print("   CUSTOM WEB-SEARCH AI AGENT (From Scratch)")
    print("🌟" * 40)
    print("\nThis agent will search the web and provide step-by-step feedback.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    # Ask user if they want streaming enabled
    stream_input = input("Enable streaming responses? (y/n, default: n): ").strip().lower()
    stream_enabled = stream_input in ['y', 'yes', 'true', '1']
    
    if stream_enabled:
        print("✅ Streaming enabled - responses will appear in real-time\n")
    else:
        print("ℹ️  Streaming disabled - full responses will appear at once\n")
    
    # Initialize agent (uses Tavily for web search)
    agent = ReActAgent(model="claude-3-5-haiku-20241022", stream=stream_enabled)
    
    while True:
        # Get user input
        user_query = input("\n💬 Your Question: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye! Thanks for using the agent.\n")
            break
        
        if not user_query:
            print("⚠️  Please enter a valid question.")
            continue
        
        try:
            # Run the agent
            answer = agent.run(user_query)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            continue


if __name__ == "__main__":
    main()
