
class message:
    def __init__(self, message, sender):
        self.message = message
        self.sender = sender
    def __str__(self):
        return f"{self.sender}: {self.message}\n"

class history:
    def __init__(self):
        self.chat_history = []
    
    def __str__(self):
        result = ""
        return result.join(str(msg) for msg in self.chat_history)

    def insert(self, message_entry):
        self.chat_history.append(message_entry)
    
x = message('hi','me')
y = message('yo', 'bot')

h = history()

h.insert(x)
h.insert(y)

print(h)
