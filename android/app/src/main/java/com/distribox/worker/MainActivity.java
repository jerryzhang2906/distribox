/*
 * Minimal DistriBox Worker Activity — starts the native Go worker binary.
 * No external dependencies (no gomobile needed for MVP).
 */
package com.distribox.worker;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.Button;
import android.graphics.Color;

public class MainActivity extends Activity {
    private TextView statusText;
    private boolean running = false;
    private Thread workerThread;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(40, 80, 40, 40);
        layout.setBackgroundColor(Color.parseColor("#1a1a2e"));

        TextView title = new TextView(this);
        title.setText("DistriBox Worker");
        title.setTextSize(24);
        title.setTextColor(Color.parseColor("#e94560"));
        layout.addView(title);

        statusText = new TextView(this);
        statusText.setText("Status: Stopped");
        statusText.setTextSize(16);
        statusText.setTextColor(Color.parseColor("#cccccc"));
        statusText.setPadding(0, 20, 0, 20);
        layout.addView(statusText);

        TextView info = new TextView(this);
        info.setText("CPU: " + android.os.Build.HARDWARE + "\n"
            + "Cores: " + Runtime.getRuntime().availableProcessors() + "\n"
            + "RAM: " + (Runtime.getRuntime().maxMemory() / 1048576) + " MB");
        info.setTextSize(13);
        info.setTextColor(Color.parseColor("#8888aa"));
        layout.addView(info);

        Button toggleBtn = new Button(this);
        toggleBtn.setText("START WORKER");
        toggleBtn.setTextSize(18);
        toggleBtn.setOnClickListener(v -> {
            if (running) {
                stopWorker();
                toggleBtn.setText("START WORKER");
                statusText.setText("Status: Stopped");
            } else {
                startWorker();
                toggleBtn.setText("STOP WORKER");
                statusText.setText("Status: Running (native)");
            }
        });
        layout.addView(toggleBtn);

        setContentView(layout);
    }

    private void startWorker() {
        running = true;
        workerThread = new Thread(() -> {
            try {
                // Extract Go worker binary from native lib dir to app files dir
                String nativeLibDir = getApplicationInfo().nativeLibraryDir;
                String workerPath = getFilesDir().getAbsolutePath() + "/distribox-worker";

                // Copy binary if not already extracted
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

                // Run the Go worker as a subprocess
                ProcessBuilder pb = new ProcessBuilder(
                    workerPath,
                    "--orchestrator", "192.168.1.100:13800",
                    "--name", android.os.Build.MODEL
                );
                pb.environment().put("HOME", getFilesDir().getAbsolutePath());
                pb.directory(getFilesDir());

                final Process proc = pb.start();
                statusText.post(() -> statusText.setText(
                    "Status: Worker running (PID " + pid(proc) + ")\nOrchestrator: 192.168.1.100:13800"));

                // Read output
                java.io.BufferedReader reader = new java.io.BufferedReader(
                    new java.io.InputStreamReader(proc.getInputStream()));
                String line;
                while (running && (line = reader.readLine()) != null) {
                    final String l = line;
                    statusText.post(() -> statusText.setText("Status: " + l));
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

    // Hack to get PID from Process (Java 9+)
    private static long pid(Process p) {
        try {
            return p.pid();
        } catch (Exception e) {
            return -1;
        }
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
