import React from 'react';
import { getURL } from '../utils/data';

export const NUM_IMAGES = 100; // number of images to display

/**
 * @param {Object} props
 * @param {tf.Tensor} props.examples
 * @param {tf.Tensor} props.predictions
 * @returns 
 */
export default function Dataset({data, predictions}) {
    console.info('Dataset', data, predictions);
    if (!data) return <div> data loading... </div>;
    const examples = data.getTestData(NUM_IMAGES);
    let imgSRC = [];
    for (let i = 0; i < examples.xs.shape[0]; i++) {
        const image = examples.xs.slice([i, 0], [1, examples.xs.shape[1]]);
        const url = getURL(image.flatten());
        imgSRC.push(url);
    }

    const labels = examples.labels.argMax(1).dataSync(); // ground truth labels

    return <div id="Dataset">
        <h2>Dataset</h2>
        {imgSRC.map((url, i) => { 
            return < figure key={i} style={{display: 'inline-block', position: 'relative', margin: '10px 5px'}}>
                <img  src={url} alt={`Example ${i}`} />
                <figcaption style={{position: 'absolute'}} >{predictions ? predictions.argMax(1).dataSync()[i] : null}</figcaption>
            </figure>
        })}
    </div>
}

