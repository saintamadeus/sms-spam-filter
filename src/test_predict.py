import sys
from predict import predict

# Test cases
test_messages = [
    "Hey, are we still meeting for lunch today?",  # Ham
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18", # Spam
    "I'm going to the store, do you need anything?", # Ham
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's", # Spam
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only." # Spam
]

print("--- Testing SMS Spam Filtering System ---")
for i, msg in enumerate(test_messages):
    print(f"\nMessage {i+1}: {msg}")
    result = predict(msg)
    print(f"Prediction: {result}")
