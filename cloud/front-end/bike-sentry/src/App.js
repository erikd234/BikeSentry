import LandingPage from "./pages/landingpage/LandingPage";
import AddNumberPage from "./pages/addnumberpage/AddNumberPage";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path='/' element={<LandingPage />} />
        <Route path='/addnumber/:towerId' element={<AddNumberPage />} />
      </Routes>
    </Router>
  );
}

export default App;
