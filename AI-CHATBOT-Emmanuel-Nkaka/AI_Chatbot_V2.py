# Rule based AI chatbot with user response

#dictionary
response =  {
    "hello" : "Hi there!",
    "hi":"Hello how are you, can i help you with anything today",
    "how are you":"I'm just fine and you?",
    "help":"Try saying hello, ask my name, or'what can you do' ",
    "what can you do":'I answer predefined questions. Type exit to leave.',
    "bye":"Goodbye! Have a great day!",
    "fine":"that's great to hear",
    "what are you":"I'm a simple rule based chatbot, with predefined responses.",
    "no":"okay",
    "thanks": "You're welcome!","good morning": "Good morning! Hope you have a wonderful day.",
    "good afternoon": "Good afternoon! How can I assist you?",
    "who made you": "I was created by a Emmanuel Nkaka a skilled programmer.",
    "who are you": "I am a simple rule-based chatbot.",
    "where are you from": "I live inside this Python program.",
    "what is your name":"i do no have a name am just a bot",
    "okay":"anything else?","what is ai": "AI stands for Artificial Intelligence.",
    "what is machine learning": "Machine learning is a branch of AI that allows computers to learn from data.",
    "what is your purpose": "My purpose is to answer predefined questions.",
    "what is chatbot": "A chatbot is software that simulates conversation.",
    "what is a chatbot": "A chatbot is software that simulates conversation.",
    "what is rule based ai": "Rule-based AI follows predefined rules and responses."
    
}

exit_commands = ("quit","bye","exit","nothing")

print("ChatBot: hi im your rule based ChatBot")
print("ChatBot: how may i assist you")

while True:
    user = input("You: ")
    clean_user = user.lower().strip()
    
    if clean_user == "":
        print("Chatbot: say something")
    elif clean_user in exit_commands:
        print("ChatBot: GoodBye!")
        break
    else:
        reply = response.get(clean_user, "sorry i do not understand")
        print("Chatbot: ",reply) 