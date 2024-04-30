import React, { useEffect, useState } from 'react';
import axios from 'axios';

function TranInfoLog() {
  const [logs, setLogs] = useState([]);

  // データをフェッチして状態を更新する関数
  const fetchLogs = async () => {
    try {
      const response = await axios.post('http://52.69.100.18:8000/read_db/', {
        table_name: 'trading_log',
        num_rows: -1 // 利用可能なすべてのデータを取得
      });
      setLogs(response.data);
    } catch (error) {
      console.error('There was an error fetching the trading logs:', error);
    }
  };

  useEffect(() => {
    fetchLogs(); // コンポーネントマウント時にデータをフェッチ
    const intervalId = setInterval(fetchLogs, 300000); // 5分ごとにデータを更新

    return () => clearInterval(intervalId); // コンポーネントのクリーンアップ
  }, []); // 空の依存配列を使用して、マウント時のみ実行

  // テーブルのスタイル設定
  const tableStyle = {
    height: '700px', // 高さ
    overflow: 'auto',
    fontSize: '0.8rem', // テキストサイズ
  };

  return (
    <div className="container mt-5">
      <div style={tableStyle}>
        <table className="table table-striped">
          <thead className="thead-dark">
            <tr>
              <th scope="col">#</th>
              <th scope="col">Date</th>
              <th scope="col">Message</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.serial}>
                <th scope="row">{log.serial}</th>
                <td>{log.date}</td>
                <td>{log.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default TranInfoLog;









