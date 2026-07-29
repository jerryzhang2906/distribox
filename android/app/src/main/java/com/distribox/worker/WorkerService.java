/*
 * DistriBox WorkerService v0.3.0
 *
 * Foreground service that manages the native Go worker binary as a child process.
 * Provides persistent notification, WakeLock for screen-off computation,
 * and a Binder interface for MainActivity status communication.
 */

package com.distribox.worker;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.concurrent.atomic.AtomicBoolean;

public class WorkerService extends Service {
    private static final String TAG = "DistriBoxWorker";
    private static final String CHANNEL_ID = "distribox_worker";
    private static final int NOTIFICATION_ID = 100;

    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread workerThread;
    private Process workerProcess;
    private PowerManager.WakeLock wakeLock;

    private volatile String orchestratorAddr = "";
    private volatile String workerStatus = "stopped";
    private volatile String lastLogLine = "";

    // ── Binder ─────────────────────────────────────────
    public class WorkerBinder extends Binder {
        public String getStatus() { return workerStatus; }
        public String getOrchestrator() { return orchestratorAddr; }
        public void setIntensity(double v) {
            Log.d(TAG, "Intensity change requested: " + v);
        }
        public void stopWorker() {
            WorkerService.this.stopWorker();
        }
    }

    private final WorkerBinder binder = new WorkerBinder();

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            return START_NOT_STICKY;
        }

        String action = intent.getAction();
        if ("STOP".equals(action)) {
            stopWorker();
            stopSelf();
            return START_NOT_STICKY;
        }

        if (running.get()) {
            Log.d(TAG, "Worker already running, ignoring start");
            return START_STICKY;
        }

        orchestratorAddr = intent.getStringExtra("orchestrator");
        if (orchestratorAddr == null) orchestratorAddr = "";

        float intensity = intent.getFloatExtra("intensity", 0.8f);
        String name = intent.getStringExtra("name");
        if (name == null) name = Build.MODEL;

        startWorker(name, orchestratorAddr, intensity);
        return START_STICKY;
    }

    // ── Start Worker ───────────────────────────────────
    private void startWorker(String name, String serverAddr, float intensity) {
        running.set(true);
        workerStatus = "starting";

        // Acquire WakeLock to keep CPU running
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        if (pm != null) {
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "DistriBox:Worker");
            wakeLock.setReferenceCounted(false);
            wakeLock.acquire(30 * 60 * 1000L); // 30 min timeout
        }

        // Show foreground notification
        String notifTitle = serverAddr.isEmpty() ?
            "Discovering via mDNS..." : "Connecting to " + serverAddr;
        startForeground(NOTIFICATION_ID, buildNotification(notifTitle, "Starting worker..."));

        // Launch worker on background thread
        workerThread = new Thread(() -> {
            try {
                runWorkerLoop(name, serverAddr, intensity);
            } catch (Exception e) {
                StringWriter sw = new StringWriter();
                e.printStackTrace(new PrintWriter(sw));
                Log.e(TAG, "Worker crashed: " + sw.toString());
                workerStatus = "error: " + e.getMessage();
            } finally {
                running.set(false);
                workerStatus = "stopped";
                updateNotification("Worker stopped", "Tap to restart");
                releaseWakeLock();
            }
        }, "distribox-worker");
        workerThread.setDaemon(true);
        workerThread.start();
    }

    private void runWorkerLoop(String name, String serverAddr, float intensity) throws Exception {
        // Extract native worker binary from assets
        String workerPath = getFilesDir().getAbsolutePath() + "/distribox-worker";
        File workerFile = new File(workerPath);

        if (!workerFile.exists() || workerFile.length() < 1000) {
            Log.i(TAG, "Extracting worker binary from assets...");
            updateNotification("Extracting binary...", "Preparing native worker");
            InputStream in = getAssets().open("distribox-worker");
            FileOutputStream out = new FileOutputStream(workerFile);
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
            Log.i(TAG, "Extracted worker binary: " + (total / 1024 / 1024) + " MB");
        }

        // Normalize server address: append default gRPC port if missing
        if (!serverAddr.isEmpty() && !serverAddr.contains(":")) {
            serverAddr = serverAddr + ":13800";
        }

        // Build process arguments
        ProcessBuilder pb;
        if (serverAddr.isEmpty()) {
            pb = new ProcessBuilder(
                workerPath,
                "--name", name,
                "--intensity", String.format(java.util.Locale.US, "%.2f", intensity)
            );
        } else {
            pb = new ProcessBuilder(
                workerPath,
                "--orchestrator", serverAddr,
                "--name", name,
                "--intensity", String.format(java.util.Locale.US, "%.2f", intensity)
            );
        }

        // Environment setup
        java.util.Map<String, String> env = pb.environment();
        env.put("HOME", getFilesDir().getAbsolutePath());
        env.put("TMPDIR", getCacheDir().getAbsolutePath());
        env.put("PATH", getFilesDir().getAbsolutePath() + ":" +
            "/system/bin:/system/xbin:/sbin");
        pb.directory(getFilesDir());
        pb.redirectErrorStream(true);

        Log.i(TAG, "Launching worker: " + String.join(" ", pb.command()));
        updateNotification("Worker running", "Connected and computing");

        final Process proc = pb.start();
        workerProcess = proc;
        workerStatus = "running";

        // Read worker output
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(proc.getInputStream()));
        String line;
        while (running.get() && (line = reader.readLine()) != null) {
            lastLogLine = line;
            Log.d(TAG, line);

            // Update notification with meaningful status lines
            if (!line.isEmpty() && line.length() < 80 && !line.startsWith(" ")) {
                updateNotification("Worker running", line);
            }
        }

        int exitCode = proc.waitFor();
        Log.i(TAG, "Worker process exited: " + exitCode);
        workerStatus = "exited:" + exitCode;
    }

    // ── Stop Worker ────────────────────────────────────
    private void stopWorker() {
        running.set(false);
        Log.i(TAG, "Stopping worker...");

        if (workerProcess != null) {
            workerProcess.destroy();
            try { workerProcess.waitFor(); } catch (Exception ignored) {}
            if (workerProcess.isAlive()) {
                workerProcess.destroyForcibly();
            }
            workerProcess = null;
        }

        if (workerThread != null) {
            workerThread.interrupt();
            workerThread = null;
        }

        releaseWakeLock();
        workerStatus = "stopped";
        stopForeground(true);
    }

    // ── Notification ───────────────────────────────────
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "DistriBox Worker",
                NotificationManager.IMPORTANCE_LOW);
            channel.setDescription("Worker service status");
            channel.setShowBadge(false);
            NotificationManager nm = getSystemService(NotificationManager.class);
            if (nm != null) nm.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification(String title, String text) {
        Intent intent = new Intent(this, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent pending = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= 23 ?
                PendingIntent.FLAG_IMMUTABLE : 0));

        Notification.Builder builder;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            builder = new Notification.Builder(this, CHANNEL_ID);
        } else {
            builder = new Notification.Builder(this);
        }

        return builder
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setContentIntent(pending)
            .setOngoing(true)
            .setPriority(Notification.PRIORITY_LOW)
            .build();
    }

    private void updateNotification(String title, String text) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) {
            nm.notify(NOTIFICATION_ID, buildNotification(title, text));
        }
    }

    // ── WakeLock ───────────────────────────────────────
    private void releaseWakeLock() {
        if (wakeLock != null && wakeLock.isHeld()) {
            try { wakeLock.release(); } catch (Exception ignored) {}
            wakeLock = null;
        }
    }

    @Override
    public void onDestroy() {
        stopWorker();
        super.onDestroy();
    }
}
