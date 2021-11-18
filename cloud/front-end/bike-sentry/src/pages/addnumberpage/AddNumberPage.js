/**
 *  Page where a user can enter their phone number and it gets added to the tower
 */

import { useState } from "react";
import { Link, useParams, useLocation } from "react-router-dom";
import axios from "../../axios";
import "./AddNumberPage.css";

function AddNumberPage(props) {
  let [text, setText] = useState("");
  const { towerId } = useParams();
  const locationHook = useLocation();
  const { name, location } = locationHook.state;
  const search = (number) => {
    if (number.length === 0) return;
    const urlSafeQuery = encodeURI(number);
    document.getElementById("addNumberInput").value = "";
    const options = {
      url: `/addnumber/${towerId}?number=${urlSafeQuery}`,
    };
    axios(options)
      .then((response) => {
        window.alert(response.data.message);
      })
      .catch((err) => {
        console.log(err);
      });
  };
  return (
    <div className='number-page-center'>
      <div className='number-page-top-margin'>
        <h1> {name} </h1>
        <h3> @ {location} </h3>
        <h4>Add your phone number to subscribe...</h4>
        <input
          id='addNumberInput'
          type='text'
          onChange={(e) => {
            setText(e.target.value);
          }}
        />
        <div className='number-page-small-margin'>
          <input
            onClick={() => {
              search(text);
            }}
            type='button'
            value='Add Number'
          />
        </div>

        <div className='top-margin'>
          <Link to='/'>Go back</Link>
        </div>
      </div>
    </div>
  );
}

export default AddNumberPage;
