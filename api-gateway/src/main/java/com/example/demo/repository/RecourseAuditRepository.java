package com.example.demo.repository;

import com.example.demo.entity.RecourseAuditEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecourseAuditRepository extends JpaRepository<RecourseAuditEntity, Long> {
}
