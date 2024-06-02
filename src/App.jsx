import './App.css';
import React, { useState, useEffect } from 'react';
import {loadData} from './utils/data';
import { train, createConvModel } from './utils/model';

import Dataset from './components/Dataset';
import Results from './components/Results';
import Train from './components/Train';

function App() {
  const [data, setData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [model, setModel] = useState(createConvModel());

  useEffect(() => {
    loadData().then((data) => {
      setData(data);
    });
  }, []);


  return (
    <div className="App">
      <header className="App-header">
        <h1>Visualization with AI</h1>
      </header>
     <div className='Body'>
      <Dataset data={data} predictions={predictions} />
      <Train model={model} data={data} setPredictions={setPredictions} setModel={setModel}/>
      </div>
    </div>
  );
};

export default App;
