/**
 * Landing page for our Bike sentry Application.
 * User gets prompted to search for their bike sentry tower.
 */
import axios from "../../axios";
import SentryCard from "./SentryCard";
import { useState, useEffect } from "react";
import "./LandingPage.css";

function LandingPage() {
  let [towerList, setTowerList] = useState([]);

  useEffect(() => {
    const options = {
      url: "/getsentrytowers",
    };
    axios(options)
      .then((response) => {
        setTowerList(response.data);
      })
      .catch(() => {
        window.alert("error");
      });
  }, []);
  return (
    <div className='center top-margin'>
      <div className='flex-item-1 center-text'>
        <img src='./shield-logo-with-bike.svg' alt='shield logo' className='shield-logo'></img>
        <h1>Welcome to Bike Sentry</h1>
        <h2>Select your tower...</h2>
        {towerList.map((tower) => {
          return (
            <SentryCard
              key={tower.tower_id}
              id={tower.tower_id}
              location={tower.tower_location}
              status={tower.status}
              name={tower.tower_name}
            />
          );
        })}
      </div>
      <div>Loading screen..</div>
    </div>
  );
}

export default LandingPage;
