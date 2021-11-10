/**
 *  Page where a user can enter their phone number and it gets added to the tower
 */

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import axios from "axios";

function AddNumberPage(props) {
  let [text, setText] = useState("");
  const { towerId } = useParams();
  const search = (query) => {
    if (query.length === 0) return;
    const urlSafeQuery = encodeURI(query);
    const options = {
      url: `http://127.0.0.1:5000/addnumber/${towerId}?number=${urlSafeQuery}`, // TODO Add the right port number
    };
    axios(options)
      .then((response) => {
        setText("");
        window.alert(response.data.message);
      })
      .catch((err) => {
        setText("");
        window.alert("failure");
        console.log(err);
      });
  };
  return (
    <div>
      <input
        type='text'
        onChange={(e) => {
          setText(e.target.value);
        }}
      />
      <input
        onClick={() => {
          search(text);
        }}
        type='button'
        value='Add Number'
      />
      <Link to='/'>Go back</Link>
    </div>
  );
}

export default AddNumberPage;
