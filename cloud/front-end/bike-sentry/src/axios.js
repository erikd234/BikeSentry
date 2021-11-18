import axios from "axios";
// Later make this a dotenv variable.
// change to false if development mode.
const production = true;
let url;
if (production === true) {
  url = "https://bike-sentry-api-2vgam74tba-uc.a.run.app";
} else {
  url = "http://127.0.0.1:5000";
}
const instance = axios.create({ baseURL: url });

export default instance;
