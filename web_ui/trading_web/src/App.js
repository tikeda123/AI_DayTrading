import React from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
// 必要なコンポーネントをインポート
import MarketDataGraph from './MarketDataGraph';
import TranInfoLog from './TranInfoLog';
import Fxtransaction from './Fxtransaction';
import AccountLog from './AccounLog';
import AccountGraph from './AccountGraph';

// 仮のAnotherComponent（実際のものに置き換えてください）
// import AnotherComponent from './AnotherComponent';

function App() {
  return (
    <Tabs
      defaultActiveKey="marketData"
      transition={false}
      id="noanim-tab-example"
      className="mb-3"
    >
      {/* MarketDataGraphコンポーネントのタブ */}
      <Tab eventKey="marketData" title="Market Data">
        <MarketDataGraph />
      </Tab>
      {/* TranInfoLogコンポーネントのタブ */}
      <Tab eventKey="transactionLog" title="Transaction Log">
        <TranInfoLog />
      </Tab>
      {/* ここに示されていない別のプログラム（コンポーネント）のタブ */}
      <Tab eventKey="fxtransaction" title="Fxtransaction Log">
        <Fxtransaction />
      </Tab>
        {/* ここに示されていない別のプログラム（コンポーネント）のタブ */}
        <Tab eventKey="accountLog" title="Account Log">
            <AccountLog />
        </Tab>
        {/* ここに示されていない別のプログラム（コンポーネント）のタブ */}
        <Tab eventKey="accountGraph" title="Account Graph">
            <AccountGraph />
        </Tab>
    </Tabs>
  );
}

export default App;









