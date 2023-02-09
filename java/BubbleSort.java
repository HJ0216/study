public class BubbleSort {
	
	public static void main(String[] args) {
		int[] ar = {78, 32, 40, 56, 25};

		for(int i=0; i<ar.length-1; i++) {
			for(int j=0; j<(ar.length-1)-i; j++) {
				if(ar[j]>ar[j+1]) {
					int tmp = ar[j+1];
					ar[j+1] = ar[j];
					ar[j] = tmp;
				} // if
			} // for j
		} // for i

		/*
		ar[0] ar[1] -> ar[0] ar[1] -> ar[0] ar[1] -> ar[0] ar[1]
		ar[1] ar[2] -> ar[1] ar[2] -> ar[1] ar[2]
		ar[2] ar[3] -> ar[2] ar[3]
		ar[3] ar[4]
		
		i = 0 -> j=0; j<4; j++
		i = 1 -> j=0; j<3; j++
		i = 2 -> j=0; j<2; j++
		i = 3 -> j=0; j<1; j++
		
		 */
		
		for(int i = 0; i<ar.length; i++) {
			System.out.print(String.format("%4d ", ar[i])); // 4자리 10진수값 지정
			// %x: 16진수, %d: 10진수, %s: String
			// System.out.printf("%4d", ar[i]);
		}
		
	}
}
