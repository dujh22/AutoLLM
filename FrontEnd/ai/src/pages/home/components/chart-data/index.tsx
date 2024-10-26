import React from 'react';
import { Card, Empty } from 'antd';
import styles from '@/pages/empty/index.module.scss';

const ChartData: React.FC = () => {
  return (
    <Card className={styles.container}>
      <div className={styles.title}>知识图谱</div>
      <Empty description="To be done..." />
    </Card>
  );
};

export default ChartData;
