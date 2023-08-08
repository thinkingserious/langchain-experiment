from langchain.utilities.twilio import TwilioAPIWrapper
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Create a new TwilioAPIWrapper object with the account SID, auth token, and from number from the environment variables
twilio = TwilioAPIWrapper(
    account_sid=os.getenv('TWILIO_ACCOUNT_SID'),
    auth_token=os.getenv('TWILIO_AUTH_TOKEN'),
    from_number=os.getenv('TWILIO_FROM_NUMBER')
)

# Use the TwilioAPIWrapper object to send a message with the text "Ahoy!" to the phone number specified in the environment variable
print(twilio.run("Ahoy!", os.getenv('TO_NUMBER')))
# You will receive a response containing your Message SID, and an SMS will be sent to you with the text 'Ahoy!'.
# https://support.twilio.com/hc/en-us/articles/223134387-What-is-a-Message-SID-
# >> SM...
