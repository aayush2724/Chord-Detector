import * as ort from 'onnxruntime-web';

const MODEL_PATH = '/model/chord_model.onnx';
const META_PATH = '/model/chord_model_meta.bin';

let session = null;
let classes = [];

export async function loadModel() {
  try {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['webgl', 'wasm'],
    });

    const metaResp = await fetch(META_PATH);
    const metaBuf = await metaResp.arrayBuffer();
    const metaView = new DataView(metaBuf);

    const classCount = metaView.getInt32(0, true);
    classes = [];
    let offset = 4;
    for (let i = 0; i < classCount; i++) {
      const len = metaView.getInt32(offset, true);
      offset += 4;
      const bytes = new Uint8Array(metaBuf, offset, len);
      classes.push(new TextDecoder().decode(bytes));
      offset += len;
    }

    return { loaded: true, classes };
  } catch (err) {
    console.error('Failed to load ONNX model:', err);
    return { loaded: false, classes: [], error: err.message };
  }
}

export function predict(landmarks) {
  if (!session || !classes.length) {
    return null;
  }

  const input = new Float32Array(landmarks);
  const tensor = new ort.Tensor('float32', input, [1, 63]);

  const results = session.run({ float_input: tensor });
  const predIdx = results.output_label.data[0]; // eslint-disable-line no-unused-vars
  const probsObj = results.output_probability[0];

  const allProbs = {};
  let maxProb = 0;
  let bestChord = classes[0];

  for (const [idx, prob] of Object.entries(probsObj)) {
    const chord = classes[parseInt(idx)];
    allProbs[chord] = prob;
    if (prob > maxProb) {
      maxProb = prob;
      bestChord = chord;
    }
  }

  return {
    chord: bestChord,
    confidence: maxProb,
    all_probs: allProbs,
    model_loaded: true,
    source: 'onnx-browser',
  };
}

export function isLoaded() {
  return session !== null && classes.length > 0;
}
