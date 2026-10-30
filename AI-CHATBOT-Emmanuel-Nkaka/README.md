# Rule-Based AI Chatbot

A simple rule-based chatbot in Python that replies to predefined user inputs. It is a "white box" program: every response can be traced from input to logic to output, with no hallucination risk because all replies are hard-coded.

## Features

- **Continuous input loop** using `while True`
- **Input sanitization** (case and whitespace handling) so `HELLO`, ` Hello ` and `hello` all match
- **Knowledge base** stored in a dictionary with 5+ intents
- **Fallback response** for unknown inputs using `dict.get()`
- **Clean exit** with `exit`, `quit` or `bye`
- **Empty-input handling** with a polite prompt

## Requirements

- Python 3.6 or newer
- No external libraries

## How to Run

1. Save the chatbot code as `chatbot.py`.
2. Open a terminal in the project folder.
3. Run:

```bash
python chatbot.py
```

## Example Session

```text
RuleBot: Hello! Type 'exit' to quit.
You: HELLO
RuleBot: Hi there!
You:   what is your name
RuleBot: I'm RuleBot, a rule-based chatbot.
You: tell me a joke
RuleBot: I do not understand.
You: exit
RuleBot: Goodbye! Have a great day!
```

## Supported Commands

| Input              | Response                                         |
| ------------------ | ------------------------------------------------ |
| `hello`            | Hi there!                                        |
| `hi`               | Hello! How can I help you?                       |
| `how are you`      | I'm just a program, but I'm running fine!        |
| `what is your name`| I'm RuleBot, a rule-based chatbot.               |
| `help`             | Try saying hello, asking my name, or ask 'what can you do'. |
| `what can you do`  | I answer predefined questions. Type exit to leave. |
| `bye` / `exit` / `quit` | Goodbye message, then the program ends      |

## How It Works

```text
User input -> Sanitize (.lower().strip()) -> Rule match? -> Yes: reply from dictionary
                                                         -> No:  fallback reply
```

| Phase           | Implementation                                   |
| --------------- | ------------------------------------------------ |
| Input loop      | `while True` runs until the kill command         |
| Sanitization    | `raw_input.lower().strip()`                      |
| Knowledge base  | `responses` dictionary (key to value lookup)     |
| Fallback        | `responses.get(clean_input, 'I do not understand.')` |
| Exit strategy   | `break` when the input is in `exit_commands`     |

A dictionary is used instead of a long `if-elif` ladder. Lookup is a direct access instead of a sequential scan, and new intents are added by adding one key-value pair.

## Project Structure

```text
.
├── chatbot.py
└── README.md
```

## Ideas for Extension

- Match keywords inside a sentence, so "hello bot" still triggers the greeting
- Strip punctuation with `string.punctuation`, so "hello!" matches
- Remember the user's name and use it in later replies
- Add nested conditions for smarter, context-aware responses
- Build a **hybrid architecture**: use rules for instant replies, and pass unmatched input to an LLM API for flexibility

## Key Skills Practiced

- Control flow and decision-making logic
- Dictionaries and hash-map lookups
- Input handling and sanitization
- Basic AI concepts (rule-based systems vs. generative models)

## License

This project is for learning purposes. Use and modify it freely.
