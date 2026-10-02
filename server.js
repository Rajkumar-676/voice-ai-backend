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

const allowedOrigins = [
  'https://voice-ai-one-iota.vercel.app',
  process.env.CLIENT_URL,
].filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (
      allowedOrigins.includes(origin) ||
      /\.vercel\.app$/.test(origin) ||
      /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
    ) {
      return callback(null, true);
    }
    // Allow any origin if CLIENT_URL is set to '*'
    if (process.env.CLIENT_URL === '*') {
      return callback(null, true);
    }
    callback(new Error(`CORS: origin ${origin} not allowed`));
  },
  credentials: true,
}));
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

// ── Root Health Check ───────────────────────────────────────────────────────
app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'Voice Assistant Backend is live! 🎙️' });
});


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

function isValidOpenRouterKey(key) {
  return key && key.startsWith('sk-or-') && key.trim().length > 10;
}

// ── Status endpoint ─────────────────────────────────────────────────────────
app.get('/api/status', (req, res) => {
  dotenv.config({ override: true });
  res.json({
    groq: isValidGroqKey(process.env.GROQ_API_KEY),
    openai: isValidOpenAiKey(process.env.OPENAI_API_KEY),
    gemini: isValidGeminiKey(process.env.GEMINI_API_KEY),
    openrouter: isValidOpenRouterKey(process.env.OPENROUTER_API_KEY),
  });
});

// ── Save API keys to .env ───────────────────────────────────────────────────
app.post('/api/save-key', (req, res) => {
  const { apiKey, geminiKey, groqKey, openrouterKey } = req.body;
  if (!apiKey && !geminiKey && !groqKey && !openrouterKey) {
    return res.status(400).json({ error: 'At least one API key is required' });
  }

  if (apiKey) process.env.OPENAI_API_KEY = apiKey;
  if (geminiKey) process.env.GEMINI_API_KEY = geminiKey;
  if (groqKey) process.env.GROQ_API_KEY = groqKey;
  if (openrouterKey) process.env.OPENROUTER_API_KEY = openrouterKey;

  const envPath = path.join(__dirname, '.env');
  const envContent = [
    `OPENAI_API_KEY=${process.env.OPENAI_API_KEY || 'sk-...'}`,
    `GEMINI_API_KEY=${process.env.GEMINI_API_KEY || ''}`,
    `GROQ_API_KEY=${process.env.GROQ_API_KEY || ''}`,
    `OPENROUTER_API_KEY=${process.env.OPENROUTER_API_KEY || ''}`,
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

// ── Helper to clean thinking tags & formatting ──────────────────────────────
function cleanAiContent(text) {
  if (!text) return '';
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .trim();
}

// ── Format messages for OpenAI / OpenRouter multimodal ──────────────────────
function formatMessagesForVision(messages) {
  return messages.map((m) => {
    if (m.image) {
      return {
        role: m.role,
        content: [
          { type: 'text', text: m.content || 'Please analyze and answer what you see in this image clearly.' },
          { type: 'image_url', image_url: { url: m.image } },
        ],
      };
    }
    return {
      role: m.role,
      content: m.content,
    };
  });
}

// ── OpenRouter chat (Supports Text & Vision) ────────────────────────────────
async function chatWithOpenRouter(messages, openrouterKey, hasImage = false) {
  const openai = new OpenAI({
    apiKey: openrouterKey,
    baseURL: 'https://openrouter.ai/api/v1',
    defaultHeaders: {
      'HTTP-Referer': 'https://voice-ai-one-iota.vercel.app',
      'X-Title': 'Voice Assistant',
    },
  });

  const formattedMessages = hasImage ? formatMessagesForVision(messages) : messages;

  const models = hasImage
    ? [
      'inclusionai/ling-3.0-flash-vl:free',
      'google/gemini-2.0-flash-001',
      'meta-llama/llama-3.2-11b-vision-instruct',
    ]
    : [
      'meta-llama/llama-3.3-70b-instruct',
      'meta-llama/llama-3.1-8b-instruct',
      'google/gemini-2.0-flash-001',
      'deepseek/deepseek-chat',
      'mistralai/mistral-7b-instruct',
    ];

  let lastErr;
  for (const model of models) {
    try {
      const response = await openai.chat.completions.create({
        model,
        messages: formattedMessages,
        temperature: 0.6,
        max_tokens: 300,
      });
      if (response.choices?.[0]?.message?.content) {
        return cleanAiContent(response.choices[0].message.content);
      }
    } catch (err) {
      console.warn(`OpenRouter model ${model} failed:`, err?.message);
      lastErr = err;
    }
  }
  throw lastErr || new Error('All OpenRouter models failed');
}

// ── Groq chat (Text only) ───────────────────────────────────────────────────
async function chatWithGroq(messages, groqKey) {
  const groq = new Groq({ apiKey: groqKey });
  const models = [
    'openai/gpt-oss-20b',
    'groq/compound-mini',
    'openai/gpt-oss-120b',
    'qwen/qwen3.6-27b',
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
  ];
  let lastErr;
  for (const model of models) {
    try {
      const response = await groq.chat.completions.create({
        model,
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        temperature: 0.6,
        max_tokens: 300,
      });
      if (response.choices?.[0]?.message?.content) {
        return cleanAiContent(response.choices[0].message.content);
      }
    } catch (err) {
      console.warn(`Groq model ${model} failed:`, err?.message);
      lastErr = err;
    }
  }
  throw lastErr || new Error('All Groq models failed');
}

// ── Gemini chat (Supports Text & Vision) ────────────────────────────────────
async function chatWithGemini(messages, geminiKey, hasImage = false) {
  const genAI = new GoogleGenerativeAI(geminiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

  if (hasImage) {
    const lastMsg = messages[messages.length - 1];
    const image = lastMsg.image;
    const match = image ? image.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/) : null;
    const mimeType = match ? match[1] : 'image/jpeg';
    const base64Data = match ? match[2] : (image || '');

    const promptText = lastMsg.content || 'Please describe and answer what you see in this image in 1-3 sentences.';
    const parts = [{ text: promptText }];
    if (base64Data) {
      parts.push({
        inlineData: {
          data: base64Data,
          mimeType,
        },
      });
    }

    const result = await model.generateContent(parts);
    return cleanAiContent(result.response.text());
  }

  const history = messages.slice(0, -1).map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));

  const chat = model.startChat({ history });
  const result = await chat.sendMessage(messages[messages.length - 1].content);
  return cleanAiContent(result.response.text());
}

// ── OpenAI chat (Supports Text & Vision) ────────────────────────────────────
async function chatWithOpenAI(messages, openaiKey, temperature = 0.7, hasImage = false) {
  const openai = new OpenAI({ apiKey: openaiKey });
  const formattedMessages = hasImage ? formatMessagesForVision(messages) : messages;
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: formattedMessages,
    temperature,
  });
  return cleanAiContent(response.choices[0].message.content);
}

// ── /api/chat ───────────────────────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  try {
    dotenv.config({ override: true });

    const clientOpenAiKey = req.headers['x-openai-key'] || req.body.apiKey;
    const clientGeminiKey = req.headers['x-gemini-key'] || req.body.geminiKey;
    const clientGroqKey = req.headers['x-groq-key'] || req.body.groqKey;
    const clientOpenRouterKey = req.headers['x-openrouter-key'] || req.body.openrouterKey;

    const openaiKey = clientOpenAiKey || process.env.OPENAI_API_KEY;
    const geminiKey = clientGeminiKey || process.env.GEMINI_API_KEY;
    const groqKey = clientGroqKey || process.env.GROQ_API_KEY;
    const openrouterKey = clientOpenRouterKey || process.env.OPENROUTER_API_KEY;

    const { messages, temperature = 0.7 } = req.body;

    const hasImage = messages && messages.some((m) => Boolean(m.image));

    // Prepend a system prompt to define agent identity and keep answers brief
    const SYSTEM_PROMPT = {
      role: 'system',
      content: hasImage
        ? 'You are an AI Voice Agent. If asked who you are or what your name is, always state clearly: "I am an Agent" (or an AI Voice Assistant). Never say you are ChatGPT or created by OpenAI. You are analyzing an image provided by the user. Give a clear, helpful, concise answer (1–3 sentences) suitable for being read aloud. Do not use markdown formatting.'
        : 'You are an AI Voice Agent. If asked who you are, what your name is, or what you are, always state clearly: "I am an Agent" (or an AI Voice Assistant). Never say you are ChatGPT or developed by OpenAI. Keep all your answers short, clear, and concise — ideally 1–3 sentences. Avoid long explanations, bullet points, or markdown formatting.',
    };
    const messagesWithSystem = [SYSTEM_PROMPT, ...messages];

    // If an image is provided, route to Vision-capable models first!
    if (hasImage) {
      // 1. OpenRouter Vision
      if (isValidOpenRouterKey(openrouterKey)) {
        try {
          const content = await chatWithOpenRouter(messagesWithSystem, openrouterKey, true);
          return res.json({ content, role: 'assistant', provider: 'openrouter-vision' });
        } catch (orErr) {
          console.warn('OpenRouter vision failed:', orErr?.message);
        }
      }

      // 2. OpenAI Vision (gpt-4o-mini)
      if (isValidOpenAiKey(openaiKey)) {
        try {
          const content = await chatWithOpenAI(messagesWithSystem, openaiKey, temperature, true);
          return res.json({ content, role: 'assistant', provider: 'openai-vision' });
        } catch (openaiErr) {
          console.warn('OpenAI vision failed:', openaiErr?.message);
        }
      }

      // 3. Gemini Vision
      if (isValidGeminiKey(geminiKey)) {
        try {
          const content = await chatWithGemini(messagesWithSystem, geminiKey, true);
          return res.json({ content, role: 'assistant', provider: 'gemini-vision' });
        } catch (geminiErr) {
          console.error('Gemini vision failed:', geminiErr?.message);
          return res.status(500).json({ error: `Vision AI error: ${geminiErr?.message}` });
        }
      }

      return res.status(400).json({
        error: 'Image scanning requires OpenRouter, OpenAI, or Gemini API Key configured in Settings.',
      });
    }

    // Standard text query:
    // 1️⃣ Try Groq first (fastest + free)
    if (isValidGroqKey(groqKey)) {
      try {
        const content = await chatWithGroq(messagesWithSystem, groqKey);
        return res.json({ content, role: 'assistant', provider: 'groq' });
      } catch (groqErr) {
        console.warn('Groq failed:', groqErr?.message);
      }
    }

    // 2️⃣ Try OpenRouter
    if (isValidOpenRouterKey(openrouterKey)) {
      try {
        const content = await chatWithOpenRouter(messagesWithSystem, openrouterKey, false);
        return res.json({ content, role: 'assistant', provider: 'openrouter' });
      } catch (orErr) {
        console.warn('OpenRouter failed:', orErr?.message);
      }
    }

    // 3️⃣ Try OpenAI
    if (isValidOpenAiKey(openaiKey)) {
      try {
        const content = await chatWithOpenAI(messagesWithSystem, openaiKey, temperature, false);
        return res.json({ content, role: 'assistant', provider: 'openai' });
      } catch (openaiErr) {
        console.warn('OpenAI failed:', openaiErr?.message);
      }
    }

    // 4️⃣ Try Gemini
    if (isValidGeminiKey(geminiKey)) {
      try {
        const content = await chatWithGemini(messagesWithSystem, geminiKey, false);
        return res.json({ content, role: 'assistant', provider: 'gemini' });
      } catch (geminiErr) {
        console.error('Gemini failed:', geminiErr?.message);
        return res.status(500).json({ error: `AI error: ${geminiErr?.message}` });
      }
    }

    // 5️⃣ No key configured
    return res.json({
      content:
        '⚠️ No AI API key configured. Please open Settings and add your OpenRouter key (sk-or-...), Groq key (gsk_...), Gemini key, or OpenAI key to get real AI answers!',
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
  console.log(`   Groq        : ${isValidGroqKey(process.env.GROQ_API_KEY) ? '✅ configured (llama-3.3-70b)' : '❌ not set'}`);
  console.log(`   OpenRouter  : ${isValidOpenRouterKey(process.env.OPENROUTER_API_KEY) ? '✅ configured (llama-3.3-70b free)' : '❌ not set'}`);
  console.log(`   OpenAI      : ${isValidOpenAiKey(process.env.OPENAI_API_KEY) ? '✅ configured (gpt-4o-mini)' : '❌ not set'}`);
  console.log(`   Gemini      : ${isValidGeminiKey(process.env.GEMINI_API_KEY) ? '✅ configured (gemini-1.5-flash)' : '❌ not set'}\n`);
});
