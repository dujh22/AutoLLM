import React from 'react';
import { Empty } from 'antd';
import styles from './index.module.scss';

const EmptyPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <Empty description="To be done..." />
    </div>
  );
};

export default EmptyPage;
