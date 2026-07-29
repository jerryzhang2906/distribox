/*
 * WorkerService.java — DistriBox foreground service (optional)
 *
 * Alternative worker path using gomobile GoBridge (JNI).
 * Keeps a persistent notification while computing.
 * Requires FOREGROUND_SERVICE permission (Android 9+).
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
import android.util.Log;

public class WorkerService extends Service {
    private static final String TAG = "DistriBoxWorker";
    private static final String CHANNEL_ID = "distribox_worker_channel";
    private static final int NOTIFICATION_ID = 101;

    private String orchestratorAddr = "";
    private String workerName = "Android Worker";
    private double intensity = 0.8;
    private volatile boolean running = false;

    public class WorkerBinder extends Binder {
        public String getStatus() {
            return running ? "connected" : "stopped";
        }
        public String getOrchestrator() { return orchestratorAddr; }
        public void setIntensity(double val) { intensity = val; }
    }

    private final IBinder binder = new WorkerBinder();

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        Log.i(TAG, "Service created");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null) {
            if ("STOP".equals(intent.getAction())) {
                stopWorker();
                stopSelf();
                return START_NOT_STICKY;
            }
            orchestratorAddr = intent.getStringExtra("orchestrator");
            if (orchestratorAddr == null) orchestratorAddr = "";
        }

        if (!running) {
            startWorker();
        }
        return START_STICKY;
    }

    private void startWorker() {
        try {
            Notification notification = buildNotification(
                "DistriBox Worker",
                orchestratorAddr.isEmpty() ? "Auto-discovering via mDNS..." : "Connecting to " + orchestratorAddr
            );

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification,
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
            } else {
                startForeground(NOTIFICATION_ID, notification);
            }

            running = true;
            workerName = Build.MODEL + " (Android)";

            new Thread(() -> {
                try {
                    // GoBridge is loaded from gomobile AAR — may not be present
                    // This path is for gomobile-based workers
                    Log.i(TAG, "Worker started — native binary mode");
                } catch (Exception e) {
                    Log.e(TAG, "Worker error: " + e.getMessage(), e);
                    running = false;
                }
            }, "worker-service").start();

        } catch (SecurityException e) {
            Log.e(TAG, "Foreground service permission denied: " + e.getMessage());
            running = false;
        } catch (Exception e) {
            Log.e(TAG, "Service start failed: " + e.getMessage(), e);
            running = false;
        }
    }

    private void stopWorker() {
        running = false;
        try {
            stopForeground(true);
        } catch (Exception e) {
            Log.w(TAG, "stopForeground: " + e.getMessage());
        }
        Log.i(TAG, "Worker stopped");
    }

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "DistriBox Worker",
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("DistriBox compute worker is active");
            channel.setShowBadge(false);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    private Notification buildNotification(String title, String text) {
        Intent intent = new Intent(this, MainActivity.class);
        int flags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            flags |= PendingIntent.FLAG_IMMUTABLE;
        }
        PendingIntent pendingIntent = PendingIntent.getActivity(
            this, 0, intent, flags
        );

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
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(Notification.PRIORITY_LOW)
            .build();
    }

    private void updateNotification(String text) {
        Notification notification = buildNotification("DistriBox Worker", text);
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.notify(NOTIFICATION_ID, notification);
        }
    }
}
