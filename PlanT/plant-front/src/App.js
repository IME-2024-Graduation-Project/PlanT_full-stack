import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
// import MyForm from './testhello';
import Test from './components/Test';
import Home from './components/Home'
import SelectCity from './components/SelectCity';
import SelectPlace from './components/SelectPlace';
import SelectEcoLevel from './components/SelectEcoLevel'
import SelectDate from './components/SelectDate';

function App() {

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/test" element={<Test />} />
        <Route path="/" element={<Home />} />
        <Route path="/city" element={<SelectCity />} />
        <Route path='/ecolevel' element={<SelectEcoLevel />} />
        <Route path="/place" element={<SelectPlace />} />
        <Route path="/date" element={<SelectDate />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
