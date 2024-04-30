import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Table from 'react-bootstrap/Table';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/read_db/', {
          table_name: "trading_log",
          num_rows: 30
        });
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data: ", error);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, 10000); // 10秒ごとにデータをフェッチ

    // コンポーネントがアンマウントされたときにインターバルをクリア
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="container mt-5">
      <h2>Trading Log</h2>
      <Table striped bordered hover>
        <thead>
          <tr>
            <th>Serial</th>
            <th>Date</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {data.map((entry) => (
            <tr key={entry.serial}>
              <td>{entry.serial}</td>
              <td>{entry.date}</td>
              <td>{entry.message}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}

export default App;

