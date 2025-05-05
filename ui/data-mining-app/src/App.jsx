import { useState, useEffect } from 'react';
import _ from 'lodash';
import './App.css';

function App() {
  // State management
  const [algorithm, setAlgorithm] = useState('apriori');
  const [dataset, setDataset] = useState([
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'butter'],
    ['bread', 'milk', 'butter'],
    ['bread', 'milk'],
  ]);
  const [datasetText, setDatasetText] = useState(
    "bread,milk,eggs\nbread,butter\nmilk,butter\nbread,milk,butter\nbread,milk"
  );
  const [minSupport, setMinSupport] = useState(0.4);
  const [minConfidence, setMinConfidence] = useState(0.6);
  const [frequentItemsets, setFrequentItemsets] = useState([]);
  const [rules, setRules] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [datasetType, setDatasetType] = useState("custom")
  // Update dataset when text changes
  const handleDatasetChange = (text) => {
    setDatasetText(text);
    try {
      const newDataset = text.trim().split('\n').map(line => 
        line.split(',').map(item => item.trim()).filter(item => item)
      );
      setDataset(newDataset);
      setError("");
    } catch (e) {
      setError("Invalid dataset format");
    }
  };

  // Run algorithm
  const runAlgorithm = async () => {
    setIsProcessing(true);
    setError("");
    
    try {
      if(datasetType === 'custom') {
        
      }
      else if (datasetType === 'movielens') {
        const response = await fetch('http://localhost:8000/apriori/default', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            frequentItemsets: frequentItemsets,
            rules: rules,
            dataset: dataset,
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setFrequentItemsets(data.frequent_itemsets || []);
        setRules(data.rules || []);
      }
    } catch (e) {
      setError(`Error processing: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Run algorithm on initial load
  useEffect(() => {
    runAlgorithm();
  }, []);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="header-title">Datamining Frequence Itemset generation</h1>
        <p className="header-subtitle">PTIT</p>
      </header>

      <main className="main-content">
        {/* Left panel - Controls */}
        <div className="control-panel">
          <h2 className="panel-title">Algorithm Configuration</h2>
          
          <div className="mb-4">
            <label className="block mb-2 font-medium text-gray-700">Select Dataset</label>
            <div className="flex gap-4 mb-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="custom"
                  checked={datasetType === 'custom'}
                  onChange={(e) => setDatasetType(e.target.value)}
                  className="mr-2 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                />
                <span className="text-gray-700">Custom Dataset</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="movielens"
                  checked={datasetType === 'movielens'}
                  onChange={(e) => setDatasetType(e.target.value)}
                  className="mr-2 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500"
                />
                <span className="text-gray-700">MovieLens Dataset</span>
              </label>
            </div>

            {datasetType === 'custom' ? (
              <div>
                <label className="block mb-2 font-medium text-gray-700">Dataset (comma-separated items, one transaction per line)</label>
                <textarea
                  value={datasetText}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  className="w-full h-40 p-2 border border-gray-300 rounded bg-white text-gray-900 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="bread,milk,eggs"
                />
              </div>
            ) : (
              <div className="p-4 bg-gray-100 rounded">
                <p className="text-sm text-gray-700">Using <a href="https://drive.google.com/file/d/1SKwn3rQbEIPUCQf-WcUQIc3ATTmEbvhY/view?usp=sharing" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">MovieLens</a> dataset with movie ratings and preferences</p>
              </div>
            )}
          </div>

          <div className="form-group">
            <label className="form-label">Select Algorithm</label>
            <div className="radio-group">
              <label className="radio-label">
                <input
                  type="radio"
                  value="apriori"
                  checked={algorithm === 'apriori'}
                  onChange={() => setAlgorithm('apriori')}
                  className="mr-2"
                />
                Apriori
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  value="fpgrowth"
                  checked={algorithm === 'fpgrowth'}
                  onChange={() => setAlgorithm('fpgrowth')}
                  className="mr-2"
                />
                FP-Growth
              </label>
            </div>
          </div>
          
          <div className="form-group">
            <label className="form-label">
              Minimum Support ({(minSupport * 100).toFixed()}%)
            </label>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={minSupport}
              onChange={(e) => setMinSupport(parseFloat(e.target.value))}
              className="range-input"
            />
            <div className="range-labels">
              <span>10%</span>
              <span>100%</span>
            </div>
          </div>
          
          <div className="form-group">
            <label className="form-label">
              Minimum Confidence ({(minConfidence * 100).toFixed()}%)
            </label>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="range-input"
            />
            <div className="range-labels">
              <span>10%</span>
              <span>100%</span>
            </div>
          </div>
          
          <button
            onClick={runAlgorithm}
            disabled={isProcessing}
            className={`action-button ${
              isProcessing ? 'action-button-disabled' : 'action-button-primary'
            }`}
          >
            {isProcessing ? 'Processing...' : 'Run Algorithm'}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {/* Right panel - Results */}
        <div className="results-panel">
          {/* Frequent itemsets */}
          <div className="results-card">
            <h2 className="card-title">Frequent Itemsets</h2>
            <div className="table-container">
              {frequentItemsets.length > 0 ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th className="table-header">Itemset</th>
                      <th className="table-header">Support</th>
                      <th className="table-header">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {frequentItemsets.map((itemset, idx) => (
                      <tr key={idx} className="border-t border-gray-200">
                        <td className="table-cell">{itemset.items.join(', ')}</td>
                        <td className="table-cell">{itemset.support.toFixed(2)}</td>
                        <td className="table-cell">{itemset.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="p-4 text-center text-gray-500">No frequent itemsets found</p>
              )}
            </div>
          </div>
          
          {/* Association rules */}
          <div className="results-card">
            <h2 className="card-title">Association Rules</h2>
            <div className="table-container">
              {rules.length > 0 ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th className="table-header">Rule</th>
                      <th className="table-header">Confidence</th>
                      <th className="table-header">Lift</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rules.map((rule, idx) => (
                      <tr key={idx} className="border-t border-gray-200">
                        <td className="table-cell">
                          {rule.antecedent.join(', ')} → {rule.consequent.join(', ')}
                        </td>
                        <td className="table-cell">{rule.confidence.toFixed(2)}</td>
                        <td className="table-cell">{rule.lift ? rule.lift.toFixed(2) : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="p-4 text-center text-gray-500">No rules found</p>
              )}
            </div>
          </div>
          
          {/* Algorithm explanation */}
          <div className="results-card">
            <h2 className="card-title">Algorithm Details</h2>
            <div className="explanation-text">
              {algorithm === 'apriori' ? (
                <div>
                  <p className="mb-3">
                    <strong>Apriori Algorithm:</strong> A classic algorithm for frequent itemset mining and association rule learning that uses breadth-first search and a hash tree structure.
                  </p>
                  <ol className="explanation-list">
                    <li>Find all frequent 1-itemsets</li>
                    <li>Generate candidate k-itemsets from frequent (k-1)-itemsets</li>
                    <li>Prune candidates with infrequent subsets</li>
                    <li>Count support for remaining candidates</li>
                    <li>Repeat until no more frequent itemsets are found</li>
                  </ol>
                </div>
              ) : (
                <div>
                  <p className="mb-3">
                    <strong>FP-Growth Algorithm:</strong> An efficient method for finding frequent itemsets without candidate generation, using a compressed data structure called FP-tree.
                  </p>
                  <ol className="explanation-list">
                    <li>Build frequency-ordered header table from transactions</li>
                    <li>Construct FP-tree with compressed representation</li>
                    <li>Mine frequent patterns recursively on conditional pattern bases</li>
                    <li>No candidate generation needed (unlike Apriori)</li>
                  </ol>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        Data warehouse & Data Mining
      </footer>
    </div>
  );
}

export default App;
