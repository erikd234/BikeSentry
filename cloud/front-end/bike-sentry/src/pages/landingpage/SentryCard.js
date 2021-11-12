import { Link } from "react-router-dom";
import "./SentryCard.css";
function SentryCard(props) {
  return (
    <Link
      to={"/addnumber/" + props.id}
      className={"no-decoration"}
      state={{ name: props.name, location: props.location }}
    >
      <div className='card-top-margin box-border-thin text-left flex card-link'>
        <div className='labels'>
          <div className='card-item label'>Name:</div>
          <div className='card-item label'>Location:</div>
          <div className='card-item label'>Status:</div>
        </div>

        <div className='info'>
          <div className='card-item info'>
            <span style={{ color: "blue" }}>{props.name}</span>
          </div>
          <div className='card-item info'>
            <span style={{ color: "blue" }}>{props.location}</span>
          </div>
          <div className='card-item info'>
            <span style={{ color: "blue" }}>{props.status}</span>
          </div>
        </div>
      </div>
    </Link>
  );
}

export default SentryCard;
