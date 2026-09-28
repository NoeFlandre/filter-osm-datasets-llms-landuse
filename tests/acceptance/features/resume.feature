Feature: Every piece of work survives interruption
  Jobs on Grid'5000 are killed at walltime, preempted or crash. Restarting must never
  lose a completed generation, redo it, or accept a torn result as complete.

  Scenario: A job killed mid-chunk resumes where it stopped
    Given a chunk of 40 sentences
    When a job is stopped after about half of them
    And a new job runs the same chunk
    Then every sentence has exactly one canonical generation
    And the second job only generated the sentences the first one had not finished

  Scenario: Interrupted and uninterrupted runs give identical labels
    Given a chunk of 40 sentences
    When a job is stopped after about half of them
    And a new job runs the same chunk
    Then the decisions equal those of an uninterrupted run

  Scenario: A corrupt part is detected and redone
    Given a chunk of 40 sentences
    And a completed run whose first part was corrupted on disk
    When a new job runs the same chunk
    Then every sentence has exactly one canonical generation
