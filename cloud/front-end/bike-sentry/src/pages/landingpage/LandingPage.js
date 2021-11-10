/**
 * Landing page for our Bike sentry Application.
 * User gets prompted to search for their bike sentry tower.
 */
import { Link } from "react-router-dom";

function LandingPage() {
  return (
    <div>
      <h2>Landing pages</h2>
      <Link to='/addnumber/tower3'>Landing Page Link</Link>
    </div>
  );
}

export default LandingPage;
