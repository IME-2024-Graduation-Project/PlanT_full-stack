import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

function SelectDate() {
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");
  
    const setChangeDate = (dates) => {
      const [start, end] = dates;
      setStartDate(start);
      setEndDate(end);
    };
  
    const displaySelectedDates = () => {
      if (!startDate || !endDate) {
      return " ";
      } else {
      return `여행 날짜: ${startDate.toLocaleDateString()}부터 ${endDate.toLocaleDateString()}까지입니다.`;
      }
    };
  
    return (
      <main>
        <h1>Plan🌱</h1>
        <div className='Travel_Date'>
          <p>여행 날짜 선택</p>
          <DatePicker
            placeholderText="여행 날짜를 선택해주세요."
            selectsRange={true}
            className="datepicker"
            dateFormat="yyyy.MM.dd"
            selected={startDate}
            startDate={startDate}
            endDate={endDate}
            onChange={setChangeDate}/>
        </div>
        <div className='SelectedDate'>
          {displaySelectedDates()}
        </div>
        <Link to="/test">다음</Link>
      </main>
    );
  }
  
  export default SelectDate;