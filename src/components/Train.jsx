
import { train } from '../utils/model'
import React, { useState } from 'react';

export default function Train({model, data, setPredictions}) {
    const [trainLogs, setTrainLogs] = useState([]);
    const [status, setStatus] = useState('Model Loaded');

    const onIteration = (text, iteration, logs) => {
        if (text === 'onBatchEnd') {
            const newTrainLogs = [...trainLogs, logs];
            setTrainLogs(newTrainLogs);
        }
        else if (text === 'onEpochEnd') {
            console.log('Epoch End');
        }
    }

    const startTraining = async () => {
        setStatus('Training...');
        await train(model, data, onIteration);
        setStatus('Done!');
        setPredictions(model.predict(data.getTestData(100).xs));
    }

    const Logs = trainLogs.length>0 ? <>
            <span><b>Accuracy:</b>{trainLogs[trainLogs.length-1]['acc'].toFixed(3)}</span>
            <span><b>Loss:</b>{trainLogs[trainLogs.length-1]['loss'].toFixed(3)}</span>
        </> : 
        null
    
    return <div id="Train">
        <h2>Model Training</h2>
        <button onClick={startTraining}>Click to Start Training</button>
        <div id='status'><b>Status:</b> {status}</div>
        {Logs}
       
    </div>
}