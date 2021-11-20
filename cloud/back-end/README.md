# BikeSentry

First step copy .env.dist to .env and fill in env variables

back-end proj dir. To start venv run 
1. `Set-ExecutionPolicy Unrestricted -Scope Process`
2. `.venv\scripts\activate`

To start dev mode when already in .venv console (.venv powershell)
1. `$env:FLASK_ENV = "development"`
2. `flask run`

To start in prod mode when in .venv console (cmd)
1. `$env:FLASK_ENV = "development`
2. `flask run`

To start ngrok
1. Have ngrok installed and add ngrok key to .env
2. run ngrok http 5000 (or whatever port server is on)
3. to stop when done `taskkill /f /im ngrok.exe`

To deploy make sure .env has production mode and .env variables are filled in.
(this assumed GCP SDK is set up)
1. Next up run python deploy.py 
2. paste output into terminal
3. Follow the rest of the instructions 
