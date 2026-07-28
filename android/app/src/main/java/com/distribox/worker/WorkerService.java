/*
 * WorkerService.java — DistriBox foreground service
 *
 * Runs the Go worker agent via gomobile bridge.
 * Keeps a persistent notification while computing.
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

import gobridge.GoBridge;

public class WorkerService extends Service {
    private static final String TAG = "DistriBoxWorker";
    private static final String CHANNEL_ID = "distribox_worker";
    private static final int NOTIFICATION_ID = 1;

    private String orchestratorAddr = "";
    private String clusterToken = "";
    private String workerName = "Android Worker";
    private double intensity = 0.8;
    private boolean running = false;

    public class WorkerBinder extends Binder {
        public String getStatus() {
            return GoBridge.workerStatus();
        }
        public String getOrchestrator() {
            return orchestratorAddr;
        }
        public void setIntensity(double val) {
            intensity = val;
        }
    }

    private final IBinder binder = new WorkerBinder();

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && "STOP".equals(intent.getAction())) {
            stopWorker();
            return START_NOT_STICKY;
        }

        if (!running) {
            startWorker();
        }
        return START_STICKY;
    }

    private void startWorker() {
        // Start foreground with notification
        Notification notification = buildNotification("Discovering orchestrator...");
        startForeground(NOTIFICATION_ID, notification);

        // Read config from preferences
        workerName = Build.MODEL + " (Android)";

        // Start the Go worker in background
        new Thread(() -> {
            try {
                // Load the Go bridge
                GoBridge.startWorker(orchestratorAddr, clusterToken, workerName);
                running = true;

                // Update notification
                updateNotification("Connected and computing");

                Log.i(TAG, "Worker started successfully");
            } catch (Exception e) {
                Log.e(TAG, "Worker start failed: " + e.getMessage());
                stopWorker();
            }
        }).start();
    }

    private void stopWorker() {
        running = false;
        try {
            GoBridge.stopWorker();
        } catch (Exception e) {
            Log.e(TAG, "Error stopping worker: " + e.getMessage());
        }
        stopForeground(true);
        stopSelf();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                getString(R.string.channel_name),
                NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription(getString(R.string.channel_desc));
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    private Notification buildNotification(String text) {
        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        return new Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("DistriBox Worker")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build();
    }

    private void updateNotification(String text) {
        Notification notification = buildNotification(text);
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager != null) {
            manager.notify(NOTIFICATION_ID, notification);
        }
    }
}
