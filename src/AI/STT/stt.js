import { SpeechClient } from "@google-cloud/speech";
import fs from "fs";

process.env.GOOGLE_APPLICATION_CREDENTIALS = "./GoogleAPI.json";

const speechClient = new SpeechClient();
const SAMPLE_RATE_HERTZ = 48000;
const LANGUAGE_CODE = "fr-FR";
const ALTERNATIVE_LANGUAGE_CODES = ["en-US"];
const INTERIM_RESULTS = false;
const LOG_INTERIM_RESULTS = false;

function getRecognitionConfig() {
  const config = {
    encoding: "LINEAR16",
    sampleRateHertz: SAMPLE_RATE_HERTZ,
    audioChannelCount: 1,
    languageCode: LANGUAGE_CODE,
    alternativeLanguageCodes: ALTERNATIVE_LANGUAGE_CODES,
    maxAlternatives: 1,
    enableAutomaticPunctuation: false,
  };

  if (process.env.GOOGLE_STT_MODEL) {
    config.model = process.env.GOOGLE_STT_MODEL;
  }

  return config;
}

async function warmup_stt() {
  const file = fs.readFileSync("src/AI/STT/warmup.wav");
  const audioBytes = file.toString("base64");

  const request = {
    audio: {
      content: audioBytes,
    },
    config: getRecognitionConfig(),
  };

  try {
    await speechClient.recognize(request);
    console.log("STT warm-up complete");
  } catch (error) {
    console.error("Unable to do STT warm-up :", error);
  }
}

function transcribe() {
  let finalText = "";
  let lastTranscript = "";

  const request = {
    config: getRecognitionConfig(),
    interimResults: INTERIM_RESULTS,
  };

  const recognizeStream = speechClient.streamingRecognize(request);

  const promise = new Promise((resolve, reject) => {
    recognizeStream
      .on("error", (err) => {
        console.error("Google STT error:", err);
        reject(err);
      })
      .on("data", (data) => {
        const result = data.results?.[0];
        const transcript = result?.alternatives?.[0]?.transcript;

        if (!transcript) return;
        lastTranscript = transcript;

        if (result.isFinal) {
          console.log("[FINAL]", transcript);
          finalText += transcript + " ";
        } else if (LOG_INTERIM_RESULTS) {
          console.log("[PARTIAL]", transcript);
        }
      })
      .on("end", () => {
        resolve((finalText || lastTranscript).trim());
      });
  });

  return {
    stream: recognizeStream,
    result: promise,
  };
}

export { transcribe, warmup_stt };
