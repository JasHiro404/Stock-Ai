import React, { useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

function App() {
  const [symbol, setSymbol] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [chartData, setChartData] = useState(null);

  const fetchData = async () => {
    try {
      // 🔮 Prediction API
      const predRes = await axios.get(`http://127.0.0.1:5000/api/predict/${symbol}`);
      setPrediction(predRes.data.predicted_price);

      // 📈 History API
      const histRes = await axios.get(`http://127.0.0.1:5000/api/history/${symbol}`);

      const labels = histRes.data.map(item => item.Date);
      const prices = histRes.data.map(item => item.Close);

      setChartData({
        labels: labels,
        datasets: [
          {
            label: "Stock Price",
            data: prices,
            borderColor: "blue",
            fill: false
          }
        ]
      });

    } catch (error) {
      alert("Error fetching data");
      console.error(error);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>📊 AI Stock Dashboard</h1>

      <input
        type="text"
        placeholder="Enter Stock Symbol (RELIANCE)"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        style={{ padding: "10px", marginRight: "10px" }}
      />

      <button onClick={fetchData} style={{ padding: "10px" }}>
        Predict
      </button>

      {/* Prediction */}
      {prediction && (
        <div style={{ marginTop: "20px" }}>
          <h2>🔮 Prediction</h2>
          <p><b>Last Price:</b> ₹{prediction.last_price}</p>
          <p><b>Next Day:</b> ₹{prediction.predicted_next_day_price}</p>
          <p><b>Trend:</b> {prediction.trend}</p>
          <p><b>Confidence:</b> {prediction.confidence.toFixed(2)}%</p>
        </div>
      )}

      {/* Chart */}
      {chartData && (
        <div style={{ marginTop: "30px" }}>
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}

export default App;