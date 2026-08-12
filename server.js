import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Groq from 'groq-sdk';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 5000;

app.use(cors({ origin: '*' }));
app.use(express.json());

// ── Key validators ──────────────────────────────────────────────────────────
function isValidOpenAiKey(key) {
  return key && key.trim().length > 20 && key.startsWith('sk-') && !/^sk-[a-z]+-\.+$/.test(key);
}

function isValidGeminiKey(key) {
  return key && key.trim().length > 10 && !key.includes('...');
}

function isValidGroqKey(key) {
  return key && key.startsWith('gsk_') && key.trim().length > 10;
}

// ── Status endpoint ─────────────────────────────────────────────────────────
app.get('/api/status', (req, res) => {
  res.json({
    groq:   isValidGroqKey(process.env.GROQ_API_KEY),
    openai: isValidOpenAiKey(process.env.OPENAI_API_KEY),
    gemini: isValidGeminiKey(process.env.GEMINI_API_KEY),
  });
});

// ── Save API keys to .env ───────────────────────────────────────────────────
app.post('/api/save-key', (req, res) => {
  const { apiKey, geminiKey, groqKey } = req.body;
  if (!apiKey && !geminiKey && !groqKey) {
    return res.status(400).json({ error: 'At least one API key is required' });
  }

  if (apiKey)   process.env.OPENAI_API_KEY = apiKey;
  if (geminiKey) process.env.GEMINI_API_KEY = geminiKey;
  if (groqKey)  process.env.GROQ_API_KEY   = groqKey;

  const envPath = path.join(__dirname, '.env');
  const envContent = [
    `OPENAI_API_KEY=${process.env.OPENAI_API_KEY || 'sk-proj-...'}`,
    `GEMINI_API_KEY=${process.env.GEMINI_API_KEY || ''}`,
    `GROQ_API_KEY=${process.env.GROQ_API_KEY || ''}`,
    `PORT=${process.env.PORT || 5000}`,
  ].join('\n') + '\n';

  try {
    fs.writeFileSync(envPath, envContent, 'utf-8');
    res.json({ success: true, message: 'API key(s) saved successfully!' });
  } catch (err) {
    console.error('Failed to write .env file:', err);
    res.status(500).json({ error: 'Failed to write key to .env file' });
  }
});

// ── Groq chat ───────────────────────────────────────────────────────────────
async function chatWithGroq(messages, groqKey) {
  const groq = new Groq({ apiKey: groqKey });
  const response = await groq.chat.completions.create({
    model: 'llama-3.3-70b-versatile',   // fast + capable free model
    messages,
    temperature: 0.7,
    max_tokens: 1024,
  });
  return response.choices[0].message.content;
}

// ── Gemini chat ─────────────────────────────────────────────────────────────
async function chatWithGemini(messages, geminiKey) {
  const genAI = new GoogleGenerativeAI(geminiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

  const history = messages.slice(0, -1).map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));

  const chat = model.startChat({ history });
  const result = await chat.sendMessage(messages[messages.length - 1].content);
  return result.response.text();
}

// ── OpenAI chat ─────────────────────────────────────────────────────────────
async function chatWithOpenAI(messages, openaiKey, temperature = 0.7) {
  const openai = new OpenAI({ apiKey: openaiKey });
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages,
    temperature,
  });
  return response.choices[0].message.content;
}

// ── /api/chat ───────────────────────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  try {
    const clientOpenAiKey = req.headers['x-openai-key'] || req.body.apiKey;
    const clientGeminiKey = req.headers['x-gemini-key'] || req.body.geminiKey;
    const clientGroqKey   = req.headers['x-groq-key']   || req.body.groqKey;

    const openaiKey = clientOpenAiKey || process.env.OPENAI_API_KEY;
    const geminiKey = clientGeminiKey || process.env.GEMINI_API_KEY;
    const groqKey   = clientGroqKey   || process.env.GROQ_API_KEY;

    const { messages, temperature = 0.7 } = req.body;

    // 1️⃣ Try Groq first (fastest + free)
    if (isValidGroqKey(groqKey)) {
      try {
        const content = await chatWithGroq(messages, groqKey);
        return res.json({ content, role: 'assistant', provider: 'groq' });
      } catch (groqErr) {
        console.warn('Groq failed:', groqErr?.message);
        // fall through to next provider
      }
    }

    // 2️⃣ Try OpenAI
    if (isValidOpenAiKey(openaiKey)) {
      try {
        const content = await chatWithOpenAI(messages, openaiKey, temperature);
        return res.json({ content, role: 'assistant', provider: 'openai' });
      } catch (openaiErr) {
        console.warn('OpenAI failed:', openaiErr?.message);
      }
    }

    // 3️⃣ Try Gemini
    if (isValidGeminiKey(geminiKey)) {
      try {
        const content = await chatWithGemini(messages, geminiKey);
        return res.json({ content, role: 'assistant', provider: 'gemini' });
      } catch (geminiErr) {
        console.error('Gemini failed:', geminiErr?.message);
        return res.status(500).json({ error: `AI error: ${geminiErr?.message}` });
      }
    }

    // 4️⃣ No key configured
    return res.json({
      content:
        '⚠️ No AI API key configured. Please open Settings and add your Groq key (gsk_...), Gemini key, or OpenAI key to get real AI answers!',
      role: 'assistant',
      provider: 'none',
    });
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: error?.message || 'Server error occurred.' });
  }
});

// ── /api/tts (OpenAI only) ──────────────────────────────────────────────────
app.post('/api/tts', async (req, res) => {
  try {
    const clientKey = req.headers['x-openai-key'] || req.body.apiKey;
    const apiKey = clientKey || process.env.OPENAI_API_KEY;
    const { text, voice = 'alloy' } = req.body;

    if (!isValidOpenAiKey(apiKey)) {
      return res.status(400).json({ error: 'OpenAI API Key required for TTS.' });
    }

    const openai = new OpenAI({ apiKey });
    const mp3 = await openai.audio.speech.create({ model: 'tts-1', voice, input: text });

    const buffer = Buffer.from(await mp3.arrayBuffer());
    res.set({ 'Content-Type': 'audio/mpeg', 'Content-Length': buffer.length });
    res.send(buffer);
  } catch (error) {
    console.error('TTS error:', error);
    res.status(500).json({ error: error?.message || 'Failed to generate voice response.' });
  }
});

// ── Start ───────────────────────────────────────────────────────────────────
app.listen(port, () => {
  console.log(`\n✅ Backend listening on http://localhost:${port}`);
  console.log(`   Groq   : ${isValidGroqKey(process.env.GROQ_API_KEY)   ? '✅ configured (llama-3.3-70b)' : '❌ not set'}`);
  console.log(`   OpenAI : ${isValidOpenAiKey(process.env.OPENAI_API_KEY) ? '✅ configured (gpt-4o-mini)'  : '❌ not set'}`);
  console.log(`   Gemini : ${isValidGeminiKey(process.env.GEMINI_API_KEY) ? '✅ configured (gemini-1.5-flash)' : '❌ not set'}\n`);
});
