// Aura Rust / Tauri Backend
// Zero-latency OS cursor control via enigo, listening to the Python ML WebSocket stream.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use enigo::{Button, Coordinate, Direction, Enigo, Mouse, Settings};
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_tungstenite::connect_async;

#[derive(Deserialize, Debug)]
struct TelemetryPacket {
    #[serde(rename = "type")]
    msg_type: String,
    x: f64,
    y: f64,
    click: bool,
    is_blink_locked: bool,
    abort_triggered: bool,
}

#[tokio::main]
async fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn WebSocket listener & Enigo mouse controller thread
    tokio::spawn(async move {
        let ws_url = "ws://127.0.0.1:8765";
        println!("[Aura Tauri Core] Connecting to Python ML engine at {}...", ws_url);

        let mut enigo = match Enigo::new(&Settings::default()) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("[Aura Tauri Core] Failed to initialize Enigo: {:?}", err);
                return;
            }
        };

        loop {
            if !running_clone.load(Ordering::Relaxed) {
                break;
            }

            match connect_async(ws_url).await {
                Ok((ws_stream, _)) => {
                    println!("[Aura Tauri Core] WebSocket connected! Executing OS control loop...");
                    let (_, mut read) = ws_stream.split();
                    let mut prev_click = false;

                    while let Some(msg_res) = read.next().await {
                        if let Ok(msg) = msg_res {
                            if let Ok(text) = msg.to_text() {
                                if let Ok(pkt) = serde_json::from_str::<TelemetryPacket>(text) {
                                    if pkt.msg_type == "telemetry" && !pkt.is_blink_locked && !pkt.abort_triggered {
                                        // Move cursor with zero latency
                                        let _ = enigo.move_mouse(pkt.x as i32, pkt.y as i32, Coordinate::Abs);

                                        // Click execution on rising edge
                                        if pkt.click && !prev_click {
                                            let _ = enigo.button(Button::Left, Direction::Click);
                                            println!("[Aura Tauri Core] Dispatched click at ({}, {})", pkt.x, pkt.y);
                                        }
                                        prev_click = pkt.click;
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Python server not yet started; retry after 2 seconds
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    });

    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
