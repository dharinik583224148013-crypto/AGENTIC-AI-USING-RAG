# Acme Enterprise Cloud Platform & Operations Handbook
**Document Version:** 3.4.1  
**Last Updated:** October 2026  
**Classification:** Internal Confidential  

---

## 1. Executive Overview
Acme Enterprise provides next-generation multi-cloud orchestrator platforms for Fortune 500 enterprises. Our flagship service, **Acmeflow Core**, powers over 10 million transactions daily with 99.995% uptime guarantees.

---

## 2. Platform Architecture & Services

### 2.1 Acmeflow Core Engine
The Core Engine is built as an asynchronous microservices platform running on Kubernetes clusters. 
- **Primary Data Layer:** Distributed PostgreSQL 16 with TimescaleDB extensions for time-series metrics.
- **Caching Layer:** Redis Cluster (version 7.2) with a memory limit of 64 GB per node and LRU eviction policy.
- **Message Broker:** Apache Kafka cluster (version 3.6) with 5-node quorum handling data pipelines.
- **Max Throughput:** 45,000 requests per second per ingress cluster.
- **Global Deployment Regions:** `us-east-1` (N. Virginia), `eu-west-1` (Dublin), and `ap-southeast-1` (Singapore).

### 2.2 Security & Authentication Protocols
- All traffic must terminate with TLS 1.3 encryption.
- User authentication requires OAuth 2.0 / OpenID Connect with multi-factor authentication (MFA) enforced.
- **API Rate Limiting:** 
  - Standard Tier: 1,000 requests per minute per tenant API key.
  - Enterprise Dedicated Tier: 10,000 requests per minute with burst capacity up to 15,000 rpm.
- **Secret Management:** HashiCorp Vault enterprise cluster rotating database credentials every 24 hours.

---

## 3. Incident Management & SLA Guidelines

### 3.1 Severity Classifications
1. **Severity 1 (Critical Outage):** Complete loss of production service affecting more than 5% of active customers.
   - Initial Response Time: Under 10 minutes.
   - Engineering Lead on call: Page duty Tier-1 SRE.
   - Status update cadence: Every 30 minutes until resolution.
2. **Severity 2 (Major Degradation):** Core functionality impaired but redundant failover operational.
   - Initial Response Time: Under 30 minutes.
   - Status update cadence: Every 2 hours.
3. **Severity 3 (Minor Defect):** Non-critical feature bug, no customer data risk.
   - Initial Response Time: Within 1 business day.

### 3.2 Backup & Disaster Recovery (DR)
- **RPO (Recovery Point Objective):** Less than 5 minutes for transaction logs.
- **RTO (Recovery Time Objective):** Less than 30 minutes for complete multi-region failover.
- Automated daily snapshots are retained for 365 days in geo-redundant object storage with immutability locks enabled.

---

## 4. Human Resources & Workplace Policies

### 4.1 Remote Work Policy
- Employees are eligible for flexible hybrid work: up to 3 days remote per week.
- Annual home office stipend: \$1,200 USD per employee for ergonomic equipment and peripherals.
- Core collaboration hours are 10:00 AM to 3:00 PM local time zone.

### 4.2 Leave & Paid Time Off (PTO)
- Full-time team members receive 25 days of Paid Time Off per calendar year, accrued monthly at 2.08 days per month.
- Floating holidays: 3 additional days for personal or cultural celebrations.
- Parental leave: 16 weeks fully paid leave for primary caregivers, 8 weeks for secondary caregivers.

---

## 5. Expense Reimbursement Rules
- Meals during business travel are capped at \$85 USD per diem in standard cities and \$110 USD in high-cost cities.
- Airfare must be booked in economy class for domestic flights under 5 hours; business class is authorized for international flights exceeding 7 hours.
- Reimbursement claims must be submitted within 30 days of incurring the expense through the Expensify portal.
