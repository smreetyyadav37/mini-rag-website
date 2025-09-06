import React, { useState } from 'react';
import './index.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const [inputText, setInputText] = useState('');
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [processingTime, setProcessingTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleIngest = async (e) => {
    e.preventDefault();
    
    if (!inputText.trim()) {
      setError("Please paste some text to ingest.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Failed to ingest document: ${response.statusText}`);
      }

      const data = await response.json();
      alert(`Ingestion successful! Processed ${data.chunks_processed} chunks.`);
    } catch (err) {
      setError(`An error occurred during ingestion: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, { // And here
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get an answer from the API.');
      }

      const data = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
      setProcessingTime(data.processing_time);
    } catch (err) {
      setError(`An error occurred while querying: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Mini RAG App</h1>

      <div className="input-panel">
        <form onSubmit={handleIngest}>
          <textarea
            placeholder="Paste your document content here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
          ></textarea>
          <button type="submit" disabled={loading}>
            {loading ? 'Ingesting...' : 'Ingest Document'}
          </button>
        </form>
      </div>

      <div className="query-panel">
        <form onSubmit={handleQuery}>
          <input
            type="text"
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Get Answer'}
          </button>
        </form>
      </div>

      {loading && <div className="loading">Processing request...</div>}
      {error && <div className="error">{error}</div>}

      {answer && (
        <div className="answer-panel">
          <h2>Answer</h2>
          <p className="answer">{answer}</p>

          <h3>Sources</h3>
          <ul className="sources-list">
            {sources.map((source, index) => (
              <li key={index}>
                <strong>[{source.citation_id}] {source.source}</strong>
                <p>{source.content}</p>
              </li>
            ))}
          </ul>
          <p className="metrics">Request Time: {processingTime} (rough estimate)</p>
        </div>
      )}
    </div>
  );
}

export default App;