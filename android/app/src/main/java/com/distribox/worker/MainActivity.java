/*
 * DistriBox Worker Activity — starts the native Go worker binary.
 *
 * Supports both mDNS auto-discovery and manual IP configuration.
 * The Go worker binary handles mDNS when --orchestrator is omitted.
 */
package com.distribox.worker;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.graphics.Color;
import android.view.Gravity;

public class MainActivity extends Activity {
    private TextView statusText;
    private EditText serverInput;
    private SeekBar intensityBar;
    private boolean running = false;
    private Thread workerThread;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(40, 80, 40, 40);
        layout.setBackgroundColor(Color.parseColor("#1a1a2e"));

        // ── Title ─────────────────────────────────────
        TextView title = new TextView(this);
        title.setText("DistriBox Worker");
        title.setTextSize(24);
        title.setTextColor(Color.parseColor("#e94560"));
        title.setGravity(Gravity.CENTER);
        layout.addView(title);

        // ── Status ────────────────────────────────────
        statusText = new TextView(this);
        statusText.setText("Status: Stopped");
        statusText.setTextSize(14);
        statusText.setTextColor(Color.parseColor("#cccccc"));
        statusText.setPadding(0, 20, 0, 10);
        layout.addView(statusText);

        // ── Device Info ───────────────────────────────
        TextView info = new TextView(this);
        info.setText("Device: " + android.os.Build.MODEL + "\n"
            + "CPU: " + android.os.Build.HARDWARE + "\n"
            + "Cores: " + Runtime.getRuntime().availableProcessors() + "\n"
            + "RAM: " + (Runtime.getRuntime().maxMemory() / 1048576) + " MB");
        info.setTextSize(13);
        info.setTextColor(Color.parseColor("#8888aa"));
        info.setPadding(0, 0, 0, 20);
        layout.addView(info);

        // ── Server Address Input ──────────────────────
        TextView serverLabel = new TextView(this);
        serverLabel.setText("Orchestrator Address (leave empty for auto-discovery):");
        serverLabel.setTextSize(12);
        serverLabel.setTextColor(Color.parseColor("#8888aa"));
        layout.addView(serverLabel);

        serverInput = new EditText(this);
        serverInput.setHint("auto-discover via mDNS");
        serverInput.setTextSize(14);
        serverInput.setTextColor(Color.parseColor("#ffffff"));
        serverInput.setHintTextColor(Color.parseColor("#555577"));
        serverInput.setBackgroundColor(Color.parseColor("#16213e"));
        serverInput.setPadding(16, 12, 16, 12);
        serverInput.setSingleLine(true);
        layout.addView(serverInput);

        // ── Intensity Slider ──────────────────────────
        TextView intensityLabel = new TextView(this);
        intensityLabel.setText("Compute Intensity: 80%");
        intensityLabel.setTextSize(12);
        intensityLabel.setTextColor(Color.parseColor("#8888aa"));
        intensityLabel.setPadding(0, 20, 0, 0);
        layout.addView(intensityLabel);

        intensityBar = new SeekBar(this);
        intensityBar.setMax(100);
        intensityBar.setProgress(80);
        intensityBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                intensityLabel.setText("Compute Intensity: " + progress + "%");
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        layout.addView(intensityBar);

        // ── Toggle Button ─────────────────────────────
        Button toggleBtn = new Button(this);
        toggleBtn.setText("START WORKER");
        toggleBtn.setTextSize(18);
        toggleBtn.setPadding(0, 20, 0, 20);
        toggleBtn.setOnClickListener(v -> {
            if (running) {
                stopWorker();
                toggleBtn.setText("START WORKER");
                statusText.setText("Status: Stopped");
            } else {
                startWorker();
                toggleBtn.setText("STOP WORKER");
                statusText.setText("Status: Connecting...");
            }
        });
        layout.addView(toggleBtn);

        setContentView(layout);
    }

    private void startWorker() {
        running = true;
        String serverAddr = serverInput.getText().toString().trim();
        int intensity = intensityBar.getProgress();

        workerThread = new Thread(() -> {
            try {
                // Extract Go worker binary from assets to app files dir
                String workerPath = getFilesDir().getAbsolutePath() + "/distribox-worker";

                java.io.File workerFile = new java.io.File(workerPath);
                if (!workerFile.exists()) {
                    java.io.InputStream in = getAssets().open("distribox-worker");
                    java.io.FileOutputStream out = new java.io.FileOutputStream(workerFile);
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    in.close(); out.close();
                    workerFile.setExecutable(true);
                }

                // Build process args
                ProcessBuilder pb;
                if (serverAddr.isEmpty()) {
                    // Auto-discovery mode: Go worker will use mDNS to find orchestrator
                    pb = new ProcessBuilder(
                        workerPath,
                        "--name", android.os.Build.MODEL,
                        "--intensity", String.valueOf(intensity / 100.0)
                    );
                    statusText.post(() -> statusText.setText(
                        "Status: Discovering orchestrator via mDNS..."));
                } else {
                    pb = new ProcessBuilder(
                        workerPath,
                        "--orchestrator", serverAddr,
                        "--name", android.os.Build.MODEL,
                        "--intensity", String.valueOf(intensity / 100.0)
                    );
                    statusText.post(() -> statusText.setText(
                        "Status: Connecting to " + serverAddr + "..."));
                }
                pb.environment().put("HOME", getFilesDir().getAbsolutePath());
                pb.directory(getFilesDir());

                final Process proc = pb.start();

                // Read output
                java.io.BufferedReader reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(proc.getInputStream()));
                String line;
                while (running && (line = reader.readLine()) != null) {
                    final String l = line;
                    statusText.post(() -> statusText.setText(l));
                }
                proc.waitFor();
            } catch (Exception e) {
                final String err = e.getMessage();
                statusText.post(() -> statusText.setText("Error: " + err));
            }
            running = false;
        });
        workerThread.start();
    }

    private void stopWorker() {
        running = false;
        if (workerThread != null) {
            workerThread.interrupt();
            workerThread = null;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopWorker();
    }
}
