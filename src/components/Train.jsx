
import { train } from '../utils/model'
import React, { useState } from 'react';
import { NUM_IMAGES } from './Dataset';

export default function Train({model, data, setPredictions, setModel}) {
    const [trainLogs, setTrainLogs] = useState([]);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('Model Loaded');

    const onIteration = (text, progress, logs) => {
        if (text === 'onBatchEnd') {
            const newTrainLogs = [...trainLogs, logs];
            setTrainLogs(newTrainLogs);
            setProgress(progress);
        }
        else if (text === 'onEpochEnd') {
            console.log('Epoch End');
        }
    }

    const startTraining = async () => {
        setStatus('Training...');
        await train(model, data, onIteration);
        setStatus('Done!');
        setPredictions(model.predict(data.getTestData(NUM_IMAGES).xs));
        setModel(model);
    }

    const Logs = trainLogs.length>0 ? <>
            <div><b>Accuracy:</b>{trainLogs[trainLogs.length-1]['acc'].toFixed(3)}</div>
            <div><b>Loss:</b>{trainLogs[trainLogs.length-1]['loss'].toFixed(3)}</div>
        </> : 
        null
    
    return <div id="Train">
        <h2>Model Training</h2>
        <button onClick={startTraining}>Click to Start Training</button>
        <div id='status'><b>Status:</b> {status}</div>
        {Logs}
        <div id='progress'><b>Progress:</b> {(progress * 100).toFixed(1)}%</div>
       
    </div>
}