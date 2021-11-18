import { useState } from "react";
import axios from "../../axios";

function DevPage() {
  let [text, setText] = useState("");
  const triggerTheft = (password) => {
    if (password.length === 0) return;
    const urlSafeQuery = encodeURI(password);
    document.getElementById("addNumberInput").value = "";
    const body = {
      password: password,
    };
    axios
      .post("/theft_alert/T0", body)
      .then((response) => {
        window.alert(response.data.message);
      })
      .catch((err) => {
        console.log(err);
      });
  };
  const resolveTheft = (password) => {
    if (password.length === 0) return;
    const urlSafeQuery = encodeURI(password);
    document.getElementById("addNumberInput").value = "";
    const body = {
      password: password,
    };
    axios
      .post("/resolve_theft/T0", body)
      .then((response) => {
        window.alert(response.data.message);
      })
      .catch((err) => {
        console.log(err);
      });
  };
  return (
    <div className='center'>
      <div className='number-page-top-margin'>
        <label>Password:</label>
        <input
          id='addNumberInput'
          type='text'
          onChange={(e) => {
            setText(e.target.value);
          }}
        />
        <div>
          <button
            onClick={() => {
              triggerTheft(text);
            }}
          >
            Trigger Theft
          </button>
        </div>
        <div>
          <button
            onClick={() => {
              resolveTheft(text);
            }}
          >
            Resolve Theft
          </button>
        </div>
      </div>
    </div>
  );
}

export default DevPage;
