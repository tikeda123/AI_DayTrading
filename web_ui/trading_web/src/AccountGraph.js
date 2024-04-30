import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import 'chartjs-adapter-date-fns';
import { TimeScale } from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  TimeScale,
  Title,
  Tooltip,
  Legend
);

function AccountGraph() {
  const [accountData, setAccountData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/read_account_data/', {
          num_rows: -1,
        });
        setAccountData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, 300000);
    return () => clearInterval(intervalId);
  }, []);

  const data = {
    labels: accountData.map(data => data.date),
    datasets: [
      {
        label: 'Total Cash Flow',
        data: accountData.map(data => data.cash_out + data.amount),
        borderColor: 'rgb(54, 162, 235)',
        borderWidth: 2,
        tension: 0.1
      }
    ],
  };

  const options = {
    scales: {
      x: {
        type: 'time',
        time: {
          parser: 'yyyy-MM-dd\'T\'HH:mm:ss', // 日付のフォーマットを適切に設定
          tooltipFormat: 'yyyy-MM-dd HH:mm'
        },
        title: {
          display: true,
          text: 'Date'
        }
      },
      y: {
        beginAtZero: false
      }
    },
    responsive: true,
    maintainAspectRatio: false,
  };

  return (
    <div className="container mt-5">
      <div style={{ height: "500px" }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}


export default AccountGraph;

