import React from 'react';
import { getURL } from '../utils/data';


/**
 * @param {Object} props
 * @param {tf.Tensor} props.examples
 * @returns 
 */
export default function Dataset({examples}) {
    if (examples==null) return <div>'data loading...'</div>;
    let imgSRC = [];
    for (let i = 0; i < examples.xs.shape[0]; i++) {
        const image = examples.xs.slice([i, 0], [1, examples.xs.shape[1]]);
        const url = getURL(image.flatten());
        imgSRC.push(url);
    }

    return <div id="Dataset">
        <h2>Dataset</h2>
        {imgSRC.map((url, i) => { 
            return < >
                <img key={'img'+i} src={url} alt={`Example ${i}`} />
                {/* <span key={'text'+i}>{examples.labels.argMax(1).dataSync()[i]}</span> */}
            </>
        })}
    </div>
}

