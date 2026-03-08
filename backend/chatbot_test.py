from backend.loaders import load_all
from backend.chatbot import chatbot_recommendation

context = load_all()

print("\nMovie Chatbot Ready!")
print("Type a message (type 'exit' to stop)\n")

while True:

    message = input("You: ")

    if message.lower() == "exit":
        break

    result = chatbot_recommendation(
        user_id=10,
        message=message,
        context=context
    )

    print("\nBot:", result["reply"])
    print("Mood:", result["mood_display"])
    print("Genres used:", result["genres"])

    print("\nRecommended Movies:\n")

    for _, row in result["recommendations"].iterrows():
        print("-", row["title"], "|", row["genres"])

    print("\n-----------------------\n")