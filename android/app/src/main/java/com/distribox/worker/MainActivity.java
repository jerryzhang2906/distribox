/*
 * DistriBox Worker — Modern Android UI
 *
 * Cyberpunk-inspired dark theme with neon accents.
 * Connects to DistriBox orchestrator via native Go worker binary.
 * Supports mDNS auto-discovery and manual IP configuration.
 */
package com.distribox.worker;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.content.res.Configuration;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {
    private TextView statusText, logText, deviceInfoText;
    private EditText serverInput;
    private SeekBar intensityBar;
    private TextView intensityLabel;
    private Button toggleBtn;
    private View statusDot;
    private boolean running = false;
    private Thread workerThread;
    private Process workerProcess;
    private Handler handler = new Handler(Looper.getMainLooper());
    private StringBuilder logBuilder = new StringBuilder();
    private SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    // ── Colors ───────────────────────────────────────
    private static final int BG_PRIMARY   = 0xFF0A0E1A;
    private static final int BG_CARD      = 0xFF141B2D;
    private static final int BG_INPUT     = 0xFF1C2541;
    private static final int ACCENT       = 0xFF00D4FF;
    private static final int ACCENT_GREEN = 0xFF00E676;
    private static final int ACCENT_RED   = 0xFFFF3D60;
    private static final int TEXT_PRIMARY = 0xFFE8EAED;
    private static final int TEXT_SECONDARY = 0xFF8892B0;
    private static final int TEXT_MUTED   = 0xFF495670;
    private static final int DIVIDER      = 0xFF1E2A45;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);

        // ── Root scroll view ──────────────────────────
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        scrollView.setBackgroundColor(BG_PRIMARY);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(20);
        root.setPadding(pad, dp(48), pad, dp(32));

        // ── Header ────────────────────────────────────
        addHeader(root);

        // ── Status card ───────────────────────────────
        LinearLayout statusCard = card();
        statusCard.setPadding(dp(20), dp(16), dp(20), dp(16));

        LinearLayout statusRow = new LinearLayout(this);
        statusRow.setOrientation(LinearLayout.HORIZONTAL);
        statusRow.setGravity(Gravity.CENTER_VERTICAL);

        statusDot = new View(this);
        int dotSize = dp(12);
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dotSize, dotSize);
        dotParams.setMargins(0, 0, dp(10), 0);
        statusDot.setLayoutParams(dotParams);
        setDotColor(statusDot, ACCENT_RED);
        statusRow.addView(statusDot);

        statusText = new TextView(this);
        statusText.setText("Disconnected");
        statusText.setTextSize(16);
        statusText.setTextColor(TEXT_PRIMARY);
        statusText.setTypeface(null, android.graphics.Typeface.BOLD);
        statusRow.addView(statusText);

        statusCard.addView(statusRow);
        root.addView(statusCard);
        space(root, 12);

        // ── Device info card ──────────────────────────
        LinearLayout infoCard = card();
        infoCard.setPadding(dp(16), dp(14), dp(16), dp(14));

        deviceInfoText = new TextView(this);
        deviceInfoText.setTextSize(13);
        deviceInfoText.setTextColor(TEXT_SECONDARY);
        deviceInfoText.setLineSpacing(dp(4), 1f);
        deviceInfoText.setText(buildDeviceInfo());
        infoCard.addView(deviceInfoText);

        root.addView(infoCard);
        space(root, 12);

        // ── Server config card ────────────────────────
        LinearLayout configCard = card();
        configCard.setPadding(dp(16), dp(14), dp(16), dp(14));
        configCard.setOrientation(LinearLayout.VERTICAL);

        TextView configLabel = new TextView(this);
        configLabel.setText("Orchestrator Server");
        configLabel.setTextSize(12);
        configLabel.setTextColor(TEXT_MUTED);
        configLabel.setPadding(0, 0, 0, dp(8));
        configCard.addView(configLabel);

        serverInput = new EditText(this);
        serverInput.setHint("Leave empty for auto-discovery via mDNS");
        serverInput.setHintTextColor(TEXT_MUTED);
        serverInput.setTextSize(14);
        serverInput.setTextColor(TEXT_PRIMARY);
        serverInput.setBackgroundDrawable(bgRounded(BG_INPUT, dp(10), DIVIDER));
        serverInput.setPadding(dp(14), dp(12), dp(14), dp(12));
        serverInput.setSingleLine(true);
        configCard.addView(serverInput);

        // Intensity slider
        space(configCard, 8);
        intensityLabel = new TextView(this);
        intensityLabel.setText("Compute Intensity: 80%");
        intensityLabel.setTextSize(12);
        intensityLabel.setTextColor(TEXT_MUTED);
        configCard.addView(intensityLabel);

        intensityBar = new SeekBar(this);
        intensityBar.setMax(100);
        intensityBar.setProgress(80);
        LinearLayout.LayoutParams sbParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        sbParams.setMargins(0, dp(4), 0, 0);
        intensityBar.setLayoutParams(sbParams);
        intensityBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar bar, int p, boolean fromUser) {
                intensityLabel.setText("Compute Intensity: " + p + "%");
            }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        configCard.addView(intensityBar);

        root.addView(configCard);
        space(root, 20);

        // ── Toggle button ─────────────────────────────
        toggleBtn = new Button(this);
        toggleBtn.setText("START WORKER");
        toggleBtn.setTextSize(16);
        toggleBtn.setTextColor(0xFF0A0E1A);
        toggleBtn.setTypeface(null, android.graphics.Typeface.BOLD);
        toggleBtn.setBackgroundDrawable(bgRounded(ACCENT, dp(28), 0));
        toggleBtn.setPadding(0, dp(14), 0, dp(14));
        toggleBtn.setAllCaps(false);
        toggleBtn.setOnClickListener(v -> {
            if (running) stopWorker(); else startWorker();
        });
        root.addView(toggleBtn);
        space(root, 12);

        // ── Log card ──────────────────────────────────
        LinearLayout logCard = card();
        logCard.setPadding(dp(12), dp(10), dp(12), dp(10));

        TextView logLabel = new TextView(this);
        logLabel.setText("Worker Log");
        logLabel.setTextSize(12);
        logLabel.setTextColor(TEXT_MUTED);
        logLabel.setPadding(0, 0, 0, dp(6));
        logCard.addView(logLabel);

        logText = new TextView(this);
        logText.setTextSize(12);
        logText.setTextColor(TEXT_SECONDARY);
        logText.setTypeface(android.graphics.Typeface.MONOSPACE);
        logText.setMovementMethod(new ScrollingMovementMethod());
        logText.setText("Ready.\n");
        logCard.addView(logText);

        root.addView(logCard);

        scrollView.addView(root);
        setContentView(scrollView);
    }

    // ── Header ────────────────────────────────────────
    private void addHeader(LinearLayout root) {
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setGravity(Gravity.CENTER);

        // Logo icon
        TextView logo = new TextView(this);
        logo.setText("⚡");
        logo.setTextSize(40);
        logo.setGravity(Gravity.CENTER);
        header.addView(logo);

        // Title
        TextView title = new TextView(this);
        title.setText("DistriBox");
        title.setTextSize(28);
        title.setTextColor(ACCENT);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        title.setGravity(Gravity.CENTER);
        header.addView(title);

        // Subtitle
        TextView subtitle = new TextView(this);
        subtitle.setText("Distributed Virtual GPU");
        subtitle.setTextSize(13);
        subtitle.setTextColor(TEXT_MUTED);
        subtitle.setGravity(Gravity.CENTER);
        header.addView(subtitle);

        root.addView(header);
        space(root, 24);
    }

    // ── Start / Stop ──────────────────────────────────
    private void startWorker() {
        running = true;
        String serverAddr = serverInput.getText().toString().trim();
        int intensity = intensityBar.getProgress();
        logBuilder.setLength(0);

        toggleBtn.setText("STOP WORKER");
        toggleBtn.setBackgroundDrawable(bgRounded(ACCENT_RED, dp(28), 0));
        setDotColor(statusDot, ACCENT);
        pulseAnimation(statusDot);

        if (serverAddr.isEmpty()) {
            setStatus("Discovering orchestrator via mDNS...");
            appendLog("Starting mDNS discovery...");
        } else {
            setStatus("Connecting to " + serverAddr + "...");
            appendLog("Connecting to " + serverAddr);
        }

        workerThread = new Thread(() -> {
            try {
                // Extract Go worker binary from assets
                String workerPath = getFilesDir().getAbsolutePath() + "/distribox-worker";
                java.io.File workerFile = new java.io.File(workerPath);

                if (!workerFile.exists()) {
                    appendLog("Extracting worker binary...");
                    java.io.InputStream in = getAssets().open("distribox-worker");
                    java.io.FileOutputStream out = new java.io.FileOutputStream(workerFile);
                    byte[] buf = new byte[8192];
                    int n;
                    long total = 0;
                    while ((n = in.read(buf)) > 0) {
                        out.write(buf, 0, n);
                        total += n;
                    }
                    in.close();
                    out.close();
                    workerFile.setExecutable(true, false);
                    appendLog("Extracted " + (total / 1024 / 1024) + " MB");
                }

                // Build process args
                ProcessBuilder pb;
                if (serverAddr.isEmpty()) {
                    pb = new ProcessBuilder(
                        workerPath,
                        "--name", sanitizeName(android.os.Build.MODEL),
                        "--intensity", String.format(Locale.US, "%.1f", intensity / 100.0)
                    );
                } else {
                    pb = new ProcessBuilder(
                        workerPath,
                        "--orchestrator", serverAddr,
                        "--name", sanitizeName(android.os.Build.MODEL),
                        "--intensity", String.format(Locale.US, "%.1f", intensity / 100.0)
                    );
                }
                pb.environment().put("HOME", getFilesDir().getAbsolutePath());
                pb.environment().put("TMPDIR", getCacheDir().getAbsolutePath());
                pb.directory(getFilesDir());
                pb.redirectErrorStream(true);

                appendLog("Launching worker process...");
                final Process proc = pb.start();
                workerProcess = proc;

                setStatusMain("Connected ✓");
                appendLog("Worker PID: " + androidPid(proc));

                BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getInputStream()));
                String line;
                while (running && (line = reader.readLine()) != null) {
                    final String l = line;
                    appendLog(l);
                    // Update status with last meaningful line
                    if (!l.isEmpty() && !l.startsWith(" ") && l.length() < 80) {
                        handler.post(() -> statusText.setText(l));
                    }
                }

                int exitCode = proc.waitFor();
                appendLog("Process exited: " + exitCode);

            } catch (java.io.FileNotFoundException e) {
                final String err = "Worker binary not found in APK assets.\nPlease rebuild the APK with 'make android'.";
                handler.post(() -> {
                    setStatus("Error: Binary missing");
                    appendLog("ERROR: " + err);
                    Toast.makeText(MainActivity.this, err, Toast.LENGTH_LONG).show();
                });
            } catch (SecurityException e) {
                final String err = "Permission denied: " + e.getMessage();
                handler.post(() -> {
                    setStatus("Error: Permission denied");
                    appendLog("ERROR: " + err);
                });
            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                e.printStackTrace(new PrintWriter(sw));
                final String err = sw.toString();
                handler.post(() -> {
                    setStatus("Error: " + truncate(e.getMessage(), 60));
                    appendLog("ERROR: " + err);
                });
            }
            running = false;
            handler.post(() -> resetUI());
        });
        workerThread.setName("distribox-worker");
        workerThread.setDaemon(true);
        workerThread.start();
    }

    private void stopWorker() {
        running = false;
        appendLog("Stopping worker...");
        if (workerProcess != null) {
            try {
                workerProcess.destroy();
                workerProcess.waitFor();
                appendLog("Worker stopped.");
            } catch (Exception e) {
                appendLog("Stop error: " + e.getMessage());
            }
            workerProcess = null;
        }
        if (workerThread != null) {
            workerThread.interrupt();
            workerThread = null;
        }
        setStatus("Disconnected");
        resetUI();
    }

    private void resetUI() {
        toggleBtn.setText("START WORKER");
        toggleBtn.setBackgroundDrawable(bgRounded(ACCENT, dp(28), 0));
        setDotColor(statusDot, ACCENT_RED);
        statusDot.clearAnimation();
    }

    // ── UI Helpers ────────────────────────────────────
    private void setStatus(String msg) {
        handler.post(() -> statusText.setText(msg));
    }

    private void setStatusMain(String msg) {
        handler.post(() -> statusText.setText(msg));
    }

    private void appendLog(String msg) {
        String ts = timeFmt.format(new Date());
        handler.post(() -> {
            logBuilder.insert(0, ts + "  " + msg + "\n");
            // Keep ~50 lines
            String[] lines = logBuilder.toString().split("\n");
            if (lines.length > 50) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < 50; i++) sb.append(lines[i]).append("\n");
                logBuilder = sb;
            }
            logText.setText(logBuilder.toString());
        });
    }

    private void setDotColor(View dot, int color) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(color);
        dot.setBackground(d);
    }

    private void pulseAnimation(View v) {
        AlphaAnimation anim = new AlphaAnimation(0.4f, 1.0f);
        anim.setDuration(800);
        anim.setRepeatMode(Animation.REVERSE);
        anim.setRepeatCount(Animation.INFINITE);
        v.startAnimation(anim);
    }

    private LinearLayout card() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackgroundDrawable(bgRounded(BG_CARD, dp(14), DIVIDER));
        return card;
    }

    private void space(LinearLayout parent, int dp) {
        View v = new View(this);
        v.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp));
        parent.addView(v);
    }

    public static GradientDrawable bgRounded(int color, int radius, int borderColor) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.RECTANGLE);
        d.setColor(color);
        d.setCornerRadius(radius);
        if (borderColor != 0) {
            d.setStroke(1, borderColor);
        }
        return d;
    }

    // ── Helpers ───────────────────────────────────────
    private String buildDeviceInfo() {
        return "Device: " + android.os.Build.MODEL + "\n"
            + "CPU: " + android.os.Build.HARDWARE + " · "
            + Runtime.getRuntime().availableProcessors() + " cores\n"
            + "Android: " + android.os.Build.VERSION.RELEASE
            + " (SDK " + android.os.Build.VERSION.SDK_INT + ")\n"
            + "Arch: " + System.getProperty("os.arch", "unknown");
    }

    private static long androidPid(Process p) {
        try { return p.pid(); } catch (Exception e) { return -1; }
    }

    private static String sanitizeName(String s) {
        return s.replaceAll("[^a-zA-Z0-9 _\\-.]", "").trim();
    }

    private static String truncate(String s, int max) {
        if (s == null) return "null";
        if (s.length() <= max) return s;
        return s.substring(0, max) + "...";
    }

    private int dp(int px) {
        return (int)(px * getResources().getDisplayMetrics().density + 0.5f);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopWorker();
    }
}
