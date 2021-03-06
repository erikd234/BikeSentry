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

THEFT_CONTROL_PASSWORD = os.getenv("THEFT_CONTROL_PASSWORD")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
GCP_CREDENTIALS_PATH = os.getenv("GCP_CREDENTIALS_PATH")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PORT_NUMBER = os.getenv("PORT_NUMBER")
ENV = os.getenv("ENV")

twilio_client = client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

try:
    if ENV == "production":
        firebase_admin.initialize_app()
    else:
        # Use a service account
        cred = credentials.Certificate(GCP_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except:
    print("Skipping firebase Auth")

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
                                     your bike, and those around you safer! (reply to this message to unsubscribe)"
    message = twilio_client.messages \
        .create(
        body=thanks_for_subscribing_message,
        from_=TWILIO_FROM_NUMBER,
        to=number,
    )
    db.collection("tower_list").document(tower_id).set(tower_info)
    return {"status": "success", "message": f"{number_formatted} is now listening to the tower."}


# This will be called by Twilio through their webhook API (running on local needs ngrok).
@app.route("/removenumber", methods=['POST'])
def remove_number():
    # Start our TwiML response
    resp = MessagingResponse()

    # First off check that the number is valid
    number = request.form["From"]
    num_obj = phonenumbers.parse(number, "US")
    if not phonenumbers.is_valid_number(num_obj) or not phonenumbers.is_possible_number(num_obj):
        # this should not happen but it would help with errors.
        resp.message(f"{number} is an invalid number.")
        return str(resp)

    number_formatted = phonenumbers.format_number(num_obj, phonenumbers.PhoneNumberFormat.NATIONAL)
    tower_id = "T0"
    # Now up grab tower by its ID
    doc_ref = db.collection("tower_list").document(tower_id)
    doc = doc_ref.get()
    if not doc.exists:
        resp.message(f"Tower {tower_id} does not exist!")
        return str(resp)

    tower_info = doc.to_dict()
    tower_name = tower_info["tower_name"]
    phone_numbers_list = tower_info["phone_numbers"]

    if number_formatted not in phone_numbers_list:
        resp.message(f"{number_formatted} is not listening to the tower.")
        return str(resp)

    # Remove the number
    tower_info["phone_numbers"] = [i for i in phone_numbers_list if i != number_formatted]
    db.collection("tower_list").document(tower_id).set(tower_info)

    # Add a message
    resp.message(f"{number_formatted} has unsubscribed to the tower!")
    return str(resp)


# Sent from tower, to API to trigger text alerts for a given tower.
# Updates the status of given tower to "Theft Detected".
@app.route("/theft_alert/<tower_id>", methods=['POST'])
def theft_alert(tower_id):
    # Grab all the listeners for the given tower ID
    # password = request.form["password"]
    # if password != THEFT_CONTROL_PASSWORD:
    #   return {"status": "failure", "message": "Invalid Password."}
    doc = db.collection("tower_list").document(tower_id).get()
    tower_info = doc.to_dict()
    tower_name = tower_info["tower_name"]
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


@app.route("/resolve_theft/<tower_id>", methods=['POST'])
def resolve_theft(tower_id):
    # password = request.form["password"]
    # if password != THEFT_CONTROL_PASSWORD:
    #   return {"status": "failure", "message": "Invalid Password."}
    doc = db.collection("tower_list").document(tower_id).get()

    if not doc.exists:
        return {"status": "failure", "message": "Tower {tower_id} does not exist!"}

    tower_info = doc.to_dict()
    tower_name = tower_info["tower_name"]
    tower_location = tower_info["tower_location"]
    phone_numbers = tower_info["phone_numbers"]
    theft_resolved_message = f"Theft at {tower_name} at {tower_location} has been resolved! Thanks for helping! "
    if len(phone_numbers) == 0:
        return {"message": "No phone numbers to send messages too :("}

    for number in phone_numbers:
        message = twilio_client.messages \
            .create(
            body=theft_resolved_message,
            from_=TWILIO_FROM_NUMBER,
            to=number,
        )
    tower_info["status"] = "Sentry Mode"
    db.collection("tower_list").document(tower_id).set(tower_info)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(PORT_NUMBER))

print("bike-sentry-api is running.")
