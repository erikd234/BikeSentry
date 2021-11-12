#
# Bike sentry back-end service.
# Handles communication with Twilio and front-end where users can add phone numbers.
#
from flask import Flask
from flask import request
from flask_cors import CORS
from flask import jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import phonenumbers
import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

TWILIO_FROM_NUMBER = "+12054967415"
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

twilio_client = client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Use a service account
cred = credentials.Certificate(GCP_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
CORS(app)


@app.route("/")
def welcome():
    return "Welcome to bike sentry API!"


# phone numbers must be sanitized and contain no special characters.
@app.route("/addnumber/<tower_id>")
def add_number(tower_id):
    # First off check that the number is valid
    number = request.args.get("number")
    num_obj = phonenumbers.parse(number, "US")
    if not phonenumbers.is_valid_number(num_obj) or not phonenumbers.is_possible_number(num_obj):
        return {"status": "failure", "message": f"{number} is an invalid number."}
    number_formatted = phonenumbers.format_number(num_obj, phonenumbers.PhoneNumberFormat.NATIONAL)

    # Now up grab tower by its ID
    doc_ref = db.collection("tower_list").document(tower_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"status": "failure", "message": f"Tower {tower_id} does not exist!"}
    tower_info = doc.to_dict()
    tower_name = tower_info["tower_name"]
    tower_location = tower_info["tower_location"]
    phone_numbers_list = tower_info["phone_numbers"]

    if number_formatted in phone_numbers_list:
        return {"status": "success", "message": f"{number_formatted} is already listening to the tower."}

    max_listener_count = 10
    if len(phone_numbers_list) > max_listener_count:
        return {"status": "failure", "message": f"Tower {tower_id} is full of listeners"}

    phone_numbers_list.append(number_formatted)
    thanks_for_subscribing_message = f"Thanks for subscribing to {tower_name} @ {tower_location}. You are making both \
                                     your bike, and those around you safer!"
    message = twilio_client.messages \
        .create(
            body=thanks_for_subscribing_message,
            from_=TWILIO_FROM_NUMBER,
            to=number,
        )
    db.collection("tower_list").document(tower_id).set(tower_info)
    return {"status": "success", "message": f"{number_formatted} is now listening to the tower."}


@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
    """Respond to incoming calls with a simple text message."""
    # Start our TwiML response
    resp = MessagingResponse()

    # Add a message
    resp.message("The Robots are coming! Head for the hills!")

    return str(resp)


# This will be called by Twilio somehow.
@app.route("/removenumber/<tower_id>")
def remove_number(tower_id):
    # First off check that the number is valid
    number = request.args.get("number")
    num_obj = phonenumbers.parse(number, "US")
    if not phonenumbers.is_valid_number(num_obj) or not phonenumbers.is_possible_number(num_obj):
        return {"status": "failure", "message": f"{number} is an invalid number."}
    number_formatted = phonenumbers.format_number(num_obj, phonenumbers.PhoneNumberFormat.NATIONAL)

    # Now up grab tower by its ID
    doc_ref = db.collection("tower_list").document(tower_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"status": "failure", "message": f"Tower {tower_id} does not exist!"}
    tower_info = doc.to_dict()
    phone_numbers_list = tower_info["phone_numbers"]

    if number_formatted not in phone_numbers_list:
        return {"status": "success", "message": f"{number_formatted} is not listening to the tower."}

    # Remove the number
    tower_info["phone_numbers"] = [i for i in phone_numbers_list if i != number_formatted]
    db.collection("tower_list").document(tower_id).set(tower_info)

    return {"status": "success", "message": f"{number_formatted} is no longer listening to the tower."}


# Sent from tower, to API to trigger text alerts for a given tower.
# Updates the status of given tower to "Theft Detected".
@app.route("/theft_alert/<tower_id>")
def theft_alert(tower_id):
    # Grab all the listeners for the given tower ID

    doc = db.collection("tower_list").document(tower_id).get()
    tower_info = doc.to_dict()
    tower_name = tower_info["Name"]
    tower_location = tower_info["tower_location"]
    tower_info["status"] = "THEFT IN PROGRESS!"
    # Update tower status!
    db.collection("tower_list").document(tower_id).set(tower_info)
    phone_numbers = tower_info["phone_numbers"]
    theft_alert_message = f"Angle Grinder detected at {tower_name} at {tower_location}, sentry turrets activating. " \
                          f"Please go check on your bike! "
    if len(phone_numbers) == 0:
        return {"message": "No phone numbers to send messages too :("}

    for number in phone_numbers:
        message = twilio_client.messages \
            .create(
                body=theft_alert_message,
                from_=TWILIO_FROM_NUMBER,
                to=number,
            )

    return {"message": "Tried to send messages to all phone numbers"}


@app.route("/resolve_theft/<tower_id>")
def resolve_theft(tower_id):
    return {"status": "success", "message": f"Theft resolved, Tower {tower_id} is back on sentry mode"}


@app.route("/getsentrytowers")
def get_sentry_towers():
    # convert tree structure of towers to a list
    docs = db.collection('tower_list').stream()
    tower_list = []
    for tower in docs:
        tower_dict = tower.to_dict()
        if "phone_numbers" in tower_dict.keys():
            del tower_dict["phone_numbers"]
        tower_list.append(tower_dict)

    return jsonify(tower_list)
