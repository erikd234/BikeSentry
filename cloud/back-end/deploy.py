# Scrip to deploy to google cloud.
# prints a command that can be pasted to deploy the project to cloud run.

import os
from dotenv import load_dotenv

load_dotenv()

THEFT_CONTROL_PASSWORD = os.getenv("THEFT_CONTROL_PASSWORD")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PORT_NUMBER = os.getenv("PORT_NUMBER")
ENV = os.getenv("ENV")

os.system(f"echo gcloud run deploy bike-sentry-api --source . --update-env-vars THEFT_CONTROL_PASSWORD={THEFT_CONTROL_PASSWORD},"
          f"TWILIO_FROM_NUMBER={TWILIO_FROM_NUMBER},TWILIO_ACCOUNT_SID={TWILIO_ACCOUNT_SID},"
          f"TWILIO_AUTH_TOKEN={TWILIO_AUTH_TOKEN},ENV={ENV},PORT_NUMBER={PORT_NUMBER}")
