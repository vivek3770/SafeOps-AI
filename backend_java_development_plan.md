# SafeOps AI - Java Backend & SQL Database Development Plan

This document outlines the detailed development plan, database schemas, Spring Boot JPA models, API routes, WebSocket configurations, simulators, and integration pathways required for **Java Backend Developers** to complete the SafeOps AI prototype.

---

## 1. Architecture Overview (Microservices Pattern)

Since the AI/ML Orchestration (LangGraph, PyTorch LSTM, ChromaDB RAG) is written in Python, the recommended industry pattern is a **Microservice Architecture**:

```
 ┌──────────────────────┐                HTTP REST               ┌────────────────────────┐
 │   Java Spring Boot   ├───────────────────────────────────────>│  Python AI/ML Service  │
 │  (Port 8080 Backend) │<───────────────────────────────────────┤   (Port 8000 Engine)   │
 └──────────┬───────────┘                Response                └────────────────────────┘
            │
      JDBC / JPA
            │
 ┌──────────▼───────────┐
 │   MySQL / Postgres   │
 │ (Port 3306 Database) │
 └──────────────────────┘
```

1. **Python AI/ML Engine (FastAPI)**: Serves strictly as a stateless compute engine, exposing a single endpoint (e.g. `POST /api/eval`) that executes our LangGraph orchestrator and returns the safety scores/compliance analysis.
2. **Java Spring Boot Backend**: Manages user authentication, plant asset state, JDBC/JPA database persistence, WebSockets, background simulators, alerts dispatching, and PDF generation.

---

## 2. Database Schema (SQL - MySQL / PostgreSQL)

Below is the DDL schema compatible with MySQL or PostgreSQL. The Java backend will map these tables using Java Persistence API (JPA) Hibernate entities.

```sql
-- 1. Plant Zones Table
CREATE TABLE zones (
    zone_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    hazard_class VARCHAR(50) NOT NULL, -- e.g., 'A', 'B', 'SAFE'
    polygon_coords TEXT NOT NULL        -- JSON string coordinates
);

-- 2. Physical Sensors Table
CREATE TABLE sensors (
    sensor_id VARCHAR(50) PRIMARY KEY,
    zone_id VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,          -- e.g., 'gas', 'temp', 'pressure'
    unit VARCHAR(10) NOT NULL,          -- e.g., 'LEL%', 'C', 'bar'
    normal_min DOUBLE DEFAULT 0.0,
    normal_max DOUBLE NOT NULL,
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE CASCADE
);

-- 3. Plant Workers Table
CREATE TABLE workers (
    worker_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    shift VARCHAR(50) NOT NULL,         -- e.g., 'MORNING', 'NIGHT'
    ppe_status VARCHAR(50) NOT NULL,     -- e.g., 'COMPLIANT', 'VIOLATION'
    current_zone VARCHAR(50),
    FOREIGN KEY (current_zone) REFERENCES zones(zone_id) ON DELETE SET NULL
);

-- 4. Work Permits Table
CREATE TABLE permits (
    permit_id VARCHAR(50) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,          -- e.g., 'HOT_WORK', 'CONFINED_SPACE'
    zone_id VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,        -- e.g., 'APPROVED', 'ACTIVE', 'SUSPENDED'
    start_time TIMESTAMP NOT NULL,
    expiry TIMESTAMP NOT NULL,
    workers_assigned TEXT NOT NULL,     -- JSON array of strings e.g. ["W_101", "W_102"]
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE CASCADE
);

-- 5. Time-Series Sensor Readings Table
CREATE TABLE sensor_readings (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    sensor_id VARCHAR(50) NOT NULL,
    value DOUBLE NOT NULL,
    anomaly_score DOUBLE DEFAULT 0.0,   -- Calculated by LSTM agent
    ts TIMESTAMP NOT NULL,
    FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id) ON DELETE CASCADE,
    INDEX idx_sensor_ts (sensor_id, ts DESC)
);

-- 6. Generated Safety Alerts Table
CREATE TABLE alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    zone_id VARCHAR(50) NOT NULL,
    risk_score DOUBLE NOT NULL,
    severity VARCHAR(50) NOT NULL,      -- e.g., 'NORMAL', 'LOW', 'MED', 'HIGH', 'CRITICAL'
    zones_affected TEXT NOT NULL,       -- JSON list
    trigger_summary TEXT NOT NULL,
    status VARCHAR(50) NOT NULL,        -- e.g., 'ACTIVE', 'RESOLVED'
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id) ON DELETE CASCADE
);

-- 7. Incident Reports Table
CREATE TABLE reports (
    incident_id VARCHAR(50) PRIMARY KEY,
    alert_id VARCHAR(50) NOT NULL,
    report_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
);
```

---

## 3. Spring Boot Setup & JPA Entity mapping

### 3.1 Maven Dependencies (`pom.xml`)
The Java backend must include the following dependencies:

```xml
<dependencies>
    <!-- Web and JPA -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-websocket</artifactId>
    </dependency>
    
    <!-- Reactive WebClient (to call Python FastAPI) -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-webflux</artifactId>
    </dependency>

    <!-- Database Drivers -->
    <dependency>
        <groupId>com.mysql</groupId>
        <artifactId>mysql-connector-j</artifactId>
        <scope>runtime</scope>
    </dependency>

    <!-- iText for PDF Report Generation -->
    <dependency>
        <groupId>com.itextpdf</groupId>
        <artifactId>itextpdf</artifactId>
        <version>5.5.13.3</version>
    </dependency>

    <!-- Twilio SDK for alerts -->
    <dependency>
        <groupId>com.twilio.sdk</groupId>
        <artifactId>twilio</artifactId>
        <version>9.12.0</version>
    </dependency>

    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>
```

### 3.2 JPA Entity mapping Example (`SensorReading.java`)
Entities should use standard JPA annotations. For JSON columns (such as `workers_assigned` or `polygon_coords`), developers can use JPA converters or store them as database `TEXT` and serialize/deserialize using Jackson `ObjectMapper`.

```java
package com.safeops.model;

import lombok.Data;
import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "sensor_readings")
@Data
public class SensorReading {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "sensor_id", nullable = false)
    private String sensorId;

    @Column(nullable = false)
    private Double value;

    @Column(name = "anomaly_score")
    private Double anomalyScore;

    @Column(nullable = false)
    private LocalDateTime ts;
}
```

---

## 4. Connecting the Python LangGraph Service

The Spring Boot backend will call the Python service dynamically. We implement a service class using Spring **WebClient** (or RESTTemplate).

### 4.1 Safety Evaluation Service Class (`SafetyService.java`)

```java
package com.safeops.service;

import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import java.util.Map;
import java.util.List;

@Service
public class SafetyService {
    private final WebClient webClient;

    public SafetyService(WebClient.Builder webClientBuilder) {
        // Points to our Python FastAPI running local docker service
        this.webClient = webClientBuilder.baseUrl("http://localhost:8000").build();
    }

    public Mono<Map> evaluateSafety(Map<String, Object> statePayload) {
        return this.webClient.post()
                .uri("/api/eval")
                .bodyValue(statePayload)
                .retrieve()
                .bodyToMono(Map.class);
    }
}
```

---

## 5. Simulators & Scheduler (Spring Task Scheduling)

Instead of Python loops, Spring Boot can run background tasks using `@Scheduled` annotations or a thread-pool task scheduler.

### 5.1 Dynamic Sensor Simulator (`SensorSimulator.java`)

```java
package com.safeops.simulator;

import com.safeops.model.SensorReading;
import com.safeops.repository.SensorReadingRepository;
import com.safeops.service.SafetyService;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.Random;

@Component
public class SensorSimulator {
    private final SensorReadingRepository readingRepository;
    private final Random random = new Random();
    private String simulationMode = "NORMAL"; // NORMAL, DRIFT, SPIKE
    private double currentGas = 18.0;

    public SensorSimulator(SensorReadingRepository readingRepository) {
        this.readingRepository = readingRepository;
    }

    public void setMode(String mode) {
        this.simulationMode = mode;
    }

    @Scheduled(fixedDelay = 5000) // Runs every 5 seconds
    public void generateReadings() {
        if ("NORMAL".equals(simulationMode)) {
            currentGas += (random.nextGaussian() * 0.05) + (18.0 - currentGas) * 0.01;
        } else if ("DRIFT".equals(simulationMode)) {
            currentGas += 0.5;
        } else if ("SPIKE".equals(simulationMode)) {
            currentGas = 38.0;
        }
        currentGas = Math.max(0.0, currentGas);

        SensorReading reading = new SensorReading();
        reading.setSensorId("GAS_Z3_001");
        reading.setValue(currentGas);
        reading.setTs(LocalDateTime.now());
        
        readingRepository.save(reading);
        
        // Trigger safety evaluation pipeline here
        // Call safetyService.evaluateSafety(...)
    }
}
```

---

## 6. Java WebSocket Configuration

Exposes real-time endpoints for the React dashboard. We implement `WebSocketConfigurer` and define handlers to broadcast updates to connected clients.

```java
package com.safeops.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {
    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new SensorWebSocketHandler(), "/ws/sensors/*").setAllowedOrigins("*");
        registry.addHandler(new AlertWebSocketHandler(), "/ws/alerts/*").setAllowedOrigins("*");
    }
}
```

---

## 7. Twilio & PDF Generator (iText)

### 7.1 SMS Dispatcher (`NotificationService.java`)
```java
package com.safeops.service;

import com.twilio.Twilio;
import com.twilio.rest.api.v2010.account.Message;
import com.twilio.type.PhoneNumber;
import org.springframework.stereotype.Service;

@Service
public class NotificationService {
    public static final String ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxx";
    public static final String AUTH_TOKEN = "your_auth_token";

    public void sendAlert(String toPhone, String text) {
        Twilio.init(ACCOUNT_SID, AUTH_TOKEN);
        Message message = Message.creator(
                new PhoneNumber(toPhone), // To
                new PhoneNumber("+1234567890"), // From (Twilio number)
                text
        ).create();
        System.out.println("Twilio SMS sent: " + message.getSid());
    }
}
```

### 7.2 iText PDF Generation (`ReportGenerator.java`)
```java
package com.safeops.service;

import com.itextpdf.text.Document;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;
import org.springframework.stereotype.Service;
import java.io.FileOutputStream;

@Service
public class ReportGenerator {
    public void generateIncidentReport(String alertId, String reportPath) {
        Document document = new Document();
        try {
            PdfWriter.getInstance(document, new FileOutputStream(reportPath));
            document.open();
            document.add(new Paragraph("🚨 SAFEOPS AI - SAFETY AUDIT COMPLIANCE REPORT"));
            document.add(new Paragraph("Alert ID: " + alertId));
            document.add(new Paragraph("Generated at: " + java.time.LocalDateTime.now()));
            document.add(new Paragraph("Compliance Violation: OISD-105 Clause 4.3 (Hot Work in presence of elevated gas)"));
            document.add(new Paragraph("Incident Precedent: Vizag 2025 Explosion similarity matches at 95%."));
            document.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

---

## 8. Multi-Container Docker Compose

We orchestrate the entire platform—Java, Python, and MySQL Database—in a single Docker Compose file:

```yaml
version: '3.8'

services:
  # 1. MySQL SQL Database
  database:
    image: mysql:8.0
    container_name: safeops-mysql
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: safeops_db
    volumes:
      - mysql-data:/var/lib/mysql
    restart: always

  # 2. Python FastAPI Agent Engine
  ai-engine:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: safeops-ai-engine
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=AIzaSyBtWTydR3sspsdr2iG3Kb8oJBqejWGmEx4
      - GEMINI_MODEL=gemini-2.5-flash
    restart: always

  # 3. Java Spring Boot Service Backend
  java-backend:
    build:
      context: ./java-backend
    container_name: safeops-java-backend
    ports:
      - "8080:8080"
    depends_on:
      - database
      - ai-engine
    environment:
      - SPRING_DATASOURCE_URL=jdbc:mysql://database:3306/safeops_db?useSSL=false
      - SPRING_DATASOURCE_USERNAME=root
      - SPRING_DATASOURCE_PASSWORD=rootpassword
    restart: always

volumes:
  mysql-data:
```
