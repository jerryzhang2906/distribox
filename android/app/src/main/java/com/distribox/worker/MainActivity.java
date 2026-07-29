/*
 * DistriBox Worker — Android App v0.3.0
 *
 * Modern Material Design cyberpunk-inspired dark theme.
 * Manages a native Go worker binary via foreground service.
 * Features: mDNS auto-discovery, compute intensity control, live log viewer.
 */

package com.distribox.worker;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.view.animation.AlphaAnimation;
import android.view.animation.Animation;
import android.view.animation.TranslateAnimation;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.LayerDrawable;
import android.net.Uri;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends Activity {
    // ── UI elements ───────────────────────────────────
    private TextView statusTitle, statusSubtitle, logText, deviceInfoText, intensityLabel;
    private EditText serverInput;
    private SeekBar intensityBar;
    private Button toggleBtn;
    private View statusDot;
    private LinearLayout logPanel;
    private ImageView headerGlow;

    // ── State ─────────────────────────────────────────
    private final AtomicBoolean running = new AtomicBoolean(false);
    private WorkerService.WorkerBinder workerBinder;
    private boolean serviceBound = false;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);
    private final StringBuilder logBuilder = new StringBuilder();

    // ── Colors (cyberpunk dark theme) ─────────────────
    private static final int BG_DEEP      = 0xFF070B19;
    private static final int BG_CARD      = 0xFF0F1630;
    private static final int BG_INPUT     = 0xFF151D3D;
    private static final int ACCENT_CYAN  = 0xFF00D4FF;
    private static final int ACCENT_GREEN = 0xFF00E676;
    private static final int ACCENT_RED   = 0xFFFF3D60;
    private static final int ACCENT_PURPLE= 0xFF7C4DFF;
    private static final int ACCENT_AMBER = 0xFFFFAB00;
    private static final int TEXT_WHITE   = 0xFFE8EAED;
    private static final int TEXT_GRAY    = 0xFF8892B0;
    private static final int TEXT_MUTED   = 0xFF4A5578;
    private static final int DIVIDER      = 0xFF1A2450;

    // ── Permission request ────────────────────────────
    private static final int REQ_NOTIFICATION = 1001;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);

        // Request notification permission on Android 13+
        if (Build.VERSION.SDK_INT >= 33) {
            if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQ_NOTIFICATION);
            }
        }

        // Build the UI
        ScrollView scrollView = new ScrollView(this);
        scrollView.setFillViewport(true);
        scrollView.setBackgroundColor(BG_DEEP);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int pad = dp(18);
        root.setPadding(pad, dp(40), pad, dp(28));

        addHeader(root);
        addStatusCard(root);
        addDeviceInfoCard(root);
        addConfigCard(root);
        addToggleButton(root);
        addLogPanel(root);

        scrollView.addView(root);
        setContentView(scrollView);

        // Bind to WorkerService
        bindService(new Intent(this, WorkerService.class), serviceConn, Context.BIND_AUTO_CREATE);
    }

    // ── Header ─────────────────────────────────────────
    private void addHeader(LinearLayout root) {
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setGravity(Gravity.CENTER);

        // Glow effect behind logo
        TextView logo = new TextView(this);
        logo.setText("⚡"); // ⚡
        logo.setTextSize(44);
        logo.setTextColor(ACCENT_CYAN);
        logo.setGravity(Gravity.CENTER);
        logo.setShadowLayer(24, 0, 0, ACCENT_CYAN);
        header.addView(logo);

        space(header, 8);

        TextView title = new TextView(this);
        title.setText("DistriBox");
        title.setTextSize(30);
        title.setTextColor(ACCENT_CYAN);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setGravity(Gravity.CENTER);
        title.setShadowLayer(16, 0, 0, 0x4400D4FF);
        header.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Distributed Virtual GPU");
        subtitle.setTextSize(13);
        subtitle.setTextColor(TEXT_MUTED);
        subtitle.setGravity(Gravity.CENTER);
        subtitle.setLetterSpacing(0.08f);
        header.addView(subtitle);

        space(header, 4);

        // Version badge
        TextView versionBadge = new TextView(this);
        versionBadge.setText("v0.3.0");
        versionBadge.setTextSize(10);
        versionBadge.setTextColor(ACCENT_CYAN);
        versionBadge.setGravity(Gravity.CENTER);
        versionBadge.setBackgroundDrawable(bgRounded(0x2200D4FF, dp(8), 0x4400D4FF));
        versionBadge.setPadding(dp(12), dp(2), dp(12), dp(2));
        LinearLayout.LayoutParams badgeParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        badgeParams.gravity = Gravity.CENTER;
        versionBadge.setLayoutParams(badgeParams);
        header.addView(versionBadge);

        root.addView(header);
        space(root, 22);
    }

    // ── Status Card ────────────────────────────────────
    private void addStatusCard(LinearLayout root) {
        LinearLayout card = card();
        card.setPadding(dp(18), dp(16), dp(18), dp(16));

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);

        // Pulsing status dot
        statusDot = new View(this);
        int dotSize = dp(14);
        LinearLayout.LayoutParams dotParams = new LinearLayout.LayoutParams(dotSize, dotSize);
        dotParams.setMargins(0, 0, dp(12), 0);
        statusDot.setLayoutParams(dotParams);
        setDotColor(ACCENT_RED);
        row.addView(statusDot);

        // Status text column
        LinearLayout textCol = new LinearLayout(this);
        textCol.setOrientation(LinearLayout.VERTICAL);

        statusTitle = new TextView(this);
        statusTitle.setText("Disconnected");
        statusTitle.setTextSize(16);
        statusTitle.setTextColor(TEXT_WHITE);
        statusTitle.setTypeface(Typeface.DEFAULT_BOLD);
        textCol.addView(statusTitle);

        statusSubtitle = new TextView(this);
        statusSubtitle.setText("Tap START to connect");
        statusSubtitle.setTextSize(12);
        statusSubtitle.setTextColor(TEXT_MUTED);
        textCol.addView(statusSubtitle);

        row.addView(textCol);
        card.addView(row);
        root.addView(card);
        space(root, 10);
    }

    // ── Device Info Card ───────────────────────────────
    private void addDeviceInfoCard(LinearLayout root) {
        LinearLayout card = card();
        card.setPadding(dp(16), dp(14), dp(16), dp(14));

        deviceInfoText = new TextView(this);
        deviceInfoText.setTextSize(12);
        deviceInfoText.setTextColor(TEXT_GRAY);
        deviceInfoText.setLineSpacing(dp(3), 1f);
        deviceInfoText.setText(buildDeviceInfo());
        card.addView(deviceInfoText);

        root.addView(card);
        space(root, 10);
    }

    // ── Config Card ────────────────────────────────────
    private void addConfigCard(LinearLayout root) {
        LinearLayout card = card();
        card.setPadding(dp(16), dp(14), dp(16), dp(6));
        card.setOrientation(LinearLayout.VERTICAL);

        // Section label with icon
        LinearLayout labelRow = new LinearLayout(this);
        labelRow.setOrientation(LinearLayout.HORIZONTAL);
        labelRow.setGravity(Gravity.CENTER_VERTICAL);

        TextView iconLabel = new TextView(this);
        iconLabel.setText("🌐"); // 🌐
        iconLabel.setTextSize(14);
        labelRow.addView(iconLabel);

        TextView configLabel = new TextView(this);
        configLabel.setText("  Orchestrator Server");
        configLabel.setTextSize(13);
        configLabel.setTextColor(TEXT_MUTED);
        configLabel.setTypeface(Typeface.DEFAULT_BOLD);
        labelRow.addView(configLabel);

        card.addView(labelRow);
        space(card, 8);

        serverInput = new EditText(this);
        serverInput.setHint("Auto-discovery via mDNS (leave empty)");
        serverInput.setHintTextColor(TEXT_MUTED);
        serverInput.setTextSize(14);
        serverInput.setTextColor(TEXT_WHITE);
        serverInput.setBackgroundDrawable(bgRounded(BG_INPUT, dp(10), DIVIDER));
        serverInput.setPadding(dp(14), dp(12), dp(14), dp(12));
        serverInput.setSingleLine(true);
        card.addView(serverInput);

        space(card, 10);

        // Intensity section
        LinearLayout intensityRow = new LinearLayout(this);
        intensityRow.setOrientation(LinearLayout.HORIZONTAL);
        intensityRow.setGravity(Gravity.CENTER_VERTICAL);

        TextView powerIcon = new TextView(this);
        powerIcon.setText("⚡");
        powerIcon.setTextSize(14);
        intensityRow.addView(powerIcon);

        intensityLabel = new TextView(this);
        intensityLabel.setText("  Compute Intensity: 80%");
        intensityLabel.setTextSize(12);
        intensityLabel.setTextColor(TEXT_MUTED);
        intensityRow.addView(intensityLabel);

        card.addView(intensityRow);

        intensityBar = new SeekBar(this);
        intensityBar.setMax(100);
        intensityBar.setProgress(80);
        LinearLayout.LayoutParams sbParams = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        sbParams.setMargins(0, dp(6), 0, dp(2));
        intensityBar.setLayoutParams(sbParams);
        // Fix: prevent ScrollView from intercepting SeekBar touch events
        intensityBar.setOnTouchListener((v, event) -> {
            v.getParent().requestDisallowInterceptTouchEvent(true);
            return false;
        });
        intensityBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar bar, int p, boolean fromUser) {
                intensityLabel.setText("  Compute Intensity: " + p + "%");
                if (workerBinder != null) {
                    workerBinder.setIntensity(p / 100.0);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar bar) {}
            @Override public void onStopTrackingTouch(SeekBar bar) {}
        });
        card.addView(intensityBar);

        root.addView(card);
        space(root, 16);
    }

    // ── Toggle Button ──────────────────────────────────
    private void addToggleButton(LinearLayout root) {
        toggleBtn = new Button(this);
        toggleBtn.setText("▶  START WORKER"); // ▶
        toggleBtn.setTextSize(15);
        toggleBtn.setTextColor(BG_DEEP);
        toggleBtn.setTypeface(Typeface.DEFAULT_BOLD);
        toggleBtn.setBackgroundDrawable(buttonBg(ACCENT_CYAN));
        toggleBtn.setPadding(0, dp(16), 0, dp(16));
        toggleBtn.setAllCaps(false);
        toggleBtn.setElevation(dp(4));
        toggleBtn.setOnClickListener(v -> {
            if (running.get()) {
                stopWorker();
            } else {
                startWorker();
            }
        });
        root.addView(toggleBtn);
        space(root, 14);
    }

    // ── Log Panel ──────────────────────────────────────
    private void addLogPanel(LinearLayout root) {
        LinearLayout card = card();
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        card.setOrientation(LinearLayout.VERTICAL);

        // Collapsible header
        LinearLayout logHeader = new LinearLayout(this);
        logHeader.setOrientation(LinearLayout.HORIZONTAL);
        logHeader.setGravity(Gravity.CENTER_VERTICAL);

        TextView logIcon = new TextView(this);
        logIcon.setText("📜"); // 📜
        logIcon.setTextSize(14);
        logHeader.addView(logIcon);

        TextView logLabel = new TextView(this);
        logLabel.setText("  Worker Log");
        logLabel.setTextSize(12);
        logLabel.setTextColor(TEXT_MUTED);
        logLabel.setTypeface(Typeface.DEFAULT_BOLD);
        logHeader.addView(logLabel);

        // Toggle log panel visibility
        logHeader.setOnClickListener(v -> {
            if (logPanel.getVisibility() == View.GONE) {
                slideDown(logPanel);
            } else {
                slideUp(logPanel);
            }
        });

        card.addView(logHeader);

        logPanel = new LinearLayout(this);
        logPanel.setOrientation(LinearLayout.VERTICAL);

        space(logPanel, 6);
        View logDivider = new View(this);
        logDivider.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 1));
        logDivider.setBackgroundColor(DIVIDER);
        logPanel.addView(logDivider);
        space(logPanel, 6);

        logText = new TextView(this);
        logText.setTextSize(11);
        logText.setTextColor(TEXT_GRAY);
        logText.setTypeface(Typeface.MONOSPACE);
        logText.setMovementMethod(new ScrollingMovementMethod());
        logText.setText("Ready.\n");
        logPanel.addView(logText);

        card.addView(logPanel);
        root.addView(card);
    }

    // ── Start / Stop Worker ────────────────────────────
    private void startWorker() {
        // Check notification permission on Android 13+
        if (Build.VERSION.SDK_INT >= 33) {
            if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Notification permission required for background service",
                    Toast.LENGTH_LONG).show();
                requestPermissions(
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQ_NOTIFICATION);
                return;
            }
        }

        running.set(true);
        String serverAddr = serverInput.getText().toString().trim();
        int intensity = intensityBar.getProgress();
        logBuilder.setLength(0);

        // Update UI
        toggleBtn.setText("■  STOP WORKER"); // ■
        toggleBtn.setBackgroundDrawable(buttonBg(ACCENT_RED));
        setStatus("Connecting", serverAddr.isEmpty() ?
            "Discovering via mDNS..." : "Connecting to " + serverAddr);
        animatePulse(statusDot, ACCENT_CYAN);
        appendLog(serverAddr.isEmpty() ?
            "Starting mDNS auto-discovery..." :
            "Connecting to orchestrator at " + serverAddr);

        // Build intent for WorkerService
        Intent intent = new Intent(this, WorkerService.class);
        intent.putExtra("orchestrator", serverAddr);
        intent.putExtra("intensity", intensity / 100.0f);
        intent.putExtra("name", sanitizeName(Build.MODEL));

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }

        // Also bind for status updates
        if (!serviceBound) {
            bindService(new Intent(this, WorkerService.class), serviceConn, Context.BIND_AUTO_CREATE);
        }

        appendLog("Worker service started");
    }

    private void stopWorker() {
        running.set(false);

        // Stop via service
        Intent intent = new Intent(this, WorkerService.class);
        intent.setAction("STOP");
        startService(intent);

        if (workerBinder != null) {
            try { workerBinder.stopWorker(); } catch (Exception ignored) {}
        }

        // Reset UI
        toggleBtn.setText("▶  START WORKER");
        toggleBtn.setBackgroundDrawable(buttonBg(ACCENT_CYAN));
        setStatus("Disconnected", "Tap START to connect");
        setDotColor(ACCENT_RED);
        stopAnimation(statusDot);
        appendLog("Worker stopped.");
    }

    // ── Service Connection ─────────────────────────────
    private final ServiceConnection serviceConn = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            workerBinder = (WorkerService.WorkerBinder) service;
            serviceBound = true;

            // Start polling for status updates
            pollWorkerStatus();

            // Update UI from service state
            refreshStatusFromService();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            workerBinder = null;
            serviceBound = false;
        }
    };

    // ── Status Polling ──────────────────────────────────
    private final Runnable statusPoller = new Runnable() {
        @Override
        public void run() {
            if (!serviceBound || workerBinder == null) {
                return;
            }
            refreshStatusFromService();
            // Keep polling while running
            if (running.get()) {
                handler.postDelayed(this, 1000);
            }
        }
    };

    private void pollWorkerStatus() {
        handler.removeCallbacks(statusPoller);
        handler.postDelayed(statusPoller, 1000);
    }

    private void refreshStatusFromService() {
        if (workerBinder == null) return;
        String svcStatus = workerBinder.getStatus();
        if ("running".equals(svcStatus)) {
            running.set(true);
            toggleBtn.setText("■  STOP WORKER");
            toggleBtn.setBackgroundDrawable(buttonBg(ACCENT_RED));
            animatePulse(statusDot, ACCENT_CYAN);
            setStatus("Connected", "Distributing to " + workerBinder.getOrchestrator());
        } else if (svcStatus.startsWith("error")) {
            setStatus("Error", svcStatus);
        } else if ("starting".equals(svcStatus)) {
            setStatus("Connecting", "Establishing gRPC stream...");
        } else if ("stopped".equals(svcStatus)) {
            running.set(false);
            toggleBtn.setText("▶  START WORKER");
            toggleBtn.setBackgroundDrawable(buttonBg(ACCENT_CYAN));
            setDotColor(ACCENT_RED);
            stopAnimation(statusDot);
            setStatus("Disconnected", "Tap START to connect");
        }
    }

    // ── UI Helpers ─────────────────────────────────────
    private void setStatus(String title, String subtitle) {
        handler.post(() -> {
            statusTitle.setText(title);
            statusSubtitle.setText(subtitle);
        });
    }

    private void appendLog(String msg) {
        String ts = timeFmt.format(new Date());
        handler.post(() -> {
            logBuilder.insert(0, ts + "  " + msg + "\n");
            // Keep ~60 lines
            String[] lines = logBuilder.toString().split("\n");
            if (lines.length > 60) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < 60; i++) sb.append(lines[i]).append("\n");
                logBuilder.setLength(0);
                logBuilder.append(sb);
            }
            logText.setText(logBuilder.toString());
        });
    }

    private void setDotColor(int color) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.OVAL);
        d.setColor(color);
        statusDot.setBackground(d);
    }

    private void animatePulse(View v, int color) {
        setDotColor(color);
        AlphaAnimation anim = new AlphaAnimation(0.3f, 1.0f);
        anim.setDuration(900);
        anim.setRepeatMode(Animation.REVERSE);
        anim.setRepeatCount(Animation.INFINITE);
        v.startAnimation(anim);
    }

    private void stopAnimation(View v) {
        v.clearAnimation();
    }

    private void slideDown(View v) {
        v.setVisibility(View.VISIBLE);
        TranslateAnimation anim = new TranslateAnimation(
            Animation.RELATIVE_TO_SELF, 0f,
            Animation.RELATIVE_TO_SELF, 0f,
            Animation.RELATIVE_TO_SELF, -1f,
            Animation.RELATIVE_TO_SELF, 0f);
        anim.setDuration(250);
        v.startAnimation(anim);
    }

    private void slideUp(View v) {
        TranslateAnimation anim = new TranslateAnimation(
            Animation.RELATIVE_TO_SELF, 0f,
            Animation.RELATIVE_TO_SELF, 0f,
            Animation.RELATIVE_TO_SELF, 0f,
            Animation.RELATIVE_TO_SELF, -1f);
        anim.setDuration(250);
        anim.setAnimationListener(new Animation.AnimationListener() {
            @Override public void onAnimationEnd(Animation a) { v.setVisibility(View.GONE); }
            @Override public void onAnimationStart(Animation a) {}
            @Override public void onAnimationRepeat(Animation a) {}
        });
        v.startAnimation(anim);
    }

    private LinearLayout card() {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackgroundDrawable(cardBg());
        card.setElevation(dp(3));
        return card;
    }

    private GradientDrawable cardBg() {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.RECTANGLE);
        d.setColor(BG_CARD);
        d.setCornerRadius(dp(14));
        d.setStroke(1, DIVIDER);
        return d;
    }

    private GradientDrawable buttonBg(int color) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.RECTANGLE);
        d.setColor(color);
        d.setCornerRadius(dp(26));
        return d;
    }

    public static GradientDrawable bgRounded(int color, int radius, int borderColor) {
        GradientDrawable d = new GradientDrawable();
        d.setShape(GradientDrawable.RECTANGLE);
        d.setColor(color);
        d.setCornerRadius(radius);
        if (borderColor != 0) d.setStroke(1, borderColor);
        return d;
    }

    private void space(LinearLayout parent, int dp) {
        View v = new View(this);
        v.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp));
        parent.addView(v);
    }

    private String buildDeviceInfo() {
        return "📱  " + Build.MODEL + "\n"
            + "💠  " + Build.HARDWARE
            + " · " + Runtime.getRuntime().availableProcessors() + " cores\n"
            + "🧠  Android " + Build.VERSION.RELEASE
            + " (SDK " + Build.VERSION.SDK_INT + ")\n"
            + "🗂  " + System.getProperty("os.arch", "unknown");
    }

    private static String sanitizeName(String s) {
        return s.replaceAll("[^a-zA-Z0-9 _\\-.]", "").trim();
    }

    private int dp(int px) {
        return (int)(px * getResources().getDisplayMetrics().density + 0.5f);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (serviceBound) {
            unbindService(serviceConn);
            serviceBound = false;
        }
    }
}
